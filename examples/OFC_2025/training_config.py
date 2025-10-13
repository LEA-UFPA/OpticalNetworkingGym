#!/usr/bin/env python3
"""
Sistema de Configuração Completo para Treinamento PPO

Este módulo centraliza TODA a configuração de treinamento, incluindo:
- Configuração de ambiente
- Hiperparâmetros
- Callbacks e logging
- Diretórios e nomes de arquivos
- Reprodutibilidade (seeds)

Uso:
    from training_config import TrainingConfig, create_training_setup
    
    # Criar configuração
    config = TrainingConfig(
        experiment_name="test_run",
        env_profile="default",
        hyperparam_profile="default"
    )
    
    # Obter tudo configurado
    env, model, callbacks = create_training_setup(config)
    
    # Treinar
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList

# Imports locais
from env_config import (
    create_vec_env,
    EnvironmentProfile,
    PROFILE_FAST,
    PROFILE_DEFAULT,
    PROFILE_INTENSIVE,
    PROFILE_LIGHT_LOAD,
    PROFILE_HEAVY_LOAD,
)
from hyperparams import (
    PPOHyperparameters,
    get_hyperparams,
    linear_schedule,
    constant_schedule
)
from callbacks import (
    BlockingRateCallback,
    FragmentationCallback,
    RewardStatsCallback,
    CustomCheckpointCallback,
    TrainingMonitorCallback
)


@dataclass
class TrainingConfig:
    """
    Configuração completa de um experimento de treinamento.
    
    Esta classe centraliza TODAS as configurações necessárias para um
    treinamento reproduzível e bem documentado.
    
    Attributes:
        experiment_name: Nome único do experimento
        env_profile: Perfil do ambiente ('fast', 'default', 'intensive', etc)
        hyperparam_profile: Perfil de hiperparâmetros
        total_timesteps: Número total de timesteps para treinar
        n_envs: Número de ambientes paralelos
        seed: Seed para reprodutibilidade
        
        # Overrides opcionais
        env_overrides: Dict com overrides de configuração do ambiente
        hyperparam_overrides: Dict com overrides de hiperparâmetros
        
        # Diretórios
        base_dir: Diretório base para outputs
        checkpoint_dir: Onde salvar checkpoints
        tensorboard_dir: Onde salvar logs do tensorboard
        
        # Callbacks e logging
        checkpoint_freq: Frequência de salvamento (em steps)
        log_freq: Frequência de logging (em steps)
        save_best_model: Se deve salvar o melhor modelo
        verbose: Nível de verbosidade (0, 1, 2)
        
        # Reprodutibilidade e metadata
        description: Descrição do experimento
        tags: Tags para organização
        notes: Notas adicionais
    """
    
    # Identificação
    experiment_name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Configurações principais
    env_profile: str = "default"
    hyperparam_profile: str = "default"
    total_timesteps: int = 1000000
    n_envs: int = 8
    vec_env_type: str = "subproc"  # 'dummy' ou 'subproc'
    seed: int = 42
    
    # Overrides
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    hyperparam_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Diretórios (serão auto-configurados se vazios)
    base_dir: str = "./experiments"
    checkpoint_dir: str = ""
    tensorboard_dir: str = ""
    logs_dir: str = ""
    
    # Callbacks
    checkpoint_freq: int = 50000
    log_freq: int = 1000
    save_best_model: bool = True
    enable_blocking_rate_callback: bool = True
    enable_fragmentation_callback: bool = True
    enable_reward_stats_callback: bool = True
    
    # Logging
    verbose: int = 1
    use_tensorboard: bool = True
    save_config_to_file: bool = True
    
    # Metadata (preenchido automaticamente)
    created_at: str = ""
    run_id: str = ""
    git_commit: str = ""
    python_version: str = ""
    
    def __post_init__(self):
        """Configura valores default e valida configuração."""
        # Gera nome do experimento se vazio
        if not self.experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.env_profile}_{self.hyperparam_profile}_{timestamp}"
        
        # Gera run_id único
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configura diretórios
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(
                self.base_dir, self.experiment_name, "checkpoints"
            )
        if not self.tensorboard_dir:
            self.tensorboard_dir = os.path.join(
                self.base_dir, self.experiment_name, "tensorboard"
            )
        if not self.logs_dir:
            self.logs_dir = os.path.join(
                self.base_dir, self.experiment_name, "logs"
            )
        
        # Metadata
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        
        # Python version
        if not self.python_version:
            import sys
            self.python_version = sys.version.split()[0]
        
        # Tenta pegar git commit
        if not self.git_commit:
            try:
                import subprocess
                result = subprocess.run(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.git_commit = result.stdout.strip()
            except:
                self.git_commit = "unknown"
        
        # Valida configuração
        self._validate()
    
    def _validate(self):
        """Valida configuração."""
        # Valida perfis
        valid_env_profiles = ['fast', 'default', 'intensive', 'light_load', 'heavy_load']
        if self.env_profile not in valid_env_profiles:
            raise ValueError(
                f"Invalid env_profile '{self.env_profile}'. "
                f"Valid options: {valid_env_profiles}"
            )
        
        valid_hyper_profiles = ['fast', 'default', 'intensive', 'high_exploration', 'stable']
        if self.hyperparam_profile not in valid_hyper_profiles:
            raise ValueError(
                f"Invalid hyperparam_profile '{self.hyperparam_profile}'. "
                f"Valid options: {valid_hyper_profiles}"
            )
        
        # Valida valores numéricos
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        
        # Valida frequências
        if self.checkpoint_freq <= 0:
            raise ValueError("checkpoint_freq must be positive")
        if self.log_freq <= 0:
            raise ValueError("log_freq must be positive")
    
    def create_directories(self):
        """Cria todos os diretórios necessários."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.use_tensorboard:
            os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        if self.verbose >= 1:
            print(f"\n📁 Diretórios criados:")
            print(f"   Checkpoints: {self.checkpoint_dir}")
            if self.use_tensorboard:
                print(f"   Tensorboard: {self.tensorboard_dir}")
            print(f"   Logs: {self.logs_dir}")
    
    def save_to_file(self, filename: Optional[str] = None):
        """
        Salva configuração em arquivo YAML.
        
        Args:
            filename: Nome do arquivo (default: config.yaml no logs_dir)
        """
        if filename is None:
            filename = os.path.join(self.logs_dir, "config.yaml")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Converte para dict
        config_dict = asdict(self)
        
        # Salva em YAML
        with open(filename, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        if self.verbose >= 1:
            print(f"✓ Configuração salva em: {filename}")
        
        return filename
    
    def save_to_json(self, filename: Optional[str] = None):
        """Salva configuração em JSON."""
        if filename is None:
            filename = os.path.join(self.logs_dir, "config.json")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return filename
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'TrainingConfig':
        """
        Carrega configuração de arquivo YAML ou JSON.
        
        Args:
            filename: Caminho do arquivo
            
        Returns:
            Objeto TrainingConfig
        """
        with open(filename, 'r') as f:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            elif filename.endswith('.json'):
                config_dict = json.load(f)
            else:
                raise ValueError("File must be .yaml, .yml, or .json")
        
        return cls(**config_dict)
    
    def print_summary(self):
        """Imprime sumário da configuração."""
        print("\n" + "="*80)
        print(f" CONFIGURAÇÃO DO EXPERIMENTO: {self.experiment_name}")
        print("="*80)
        
        print(f"\n📋 Informações Gerais:")
        print(f"   Run ID: {self.run_id}")
        print(f"   Criado em: {self.created_at}")
        if self.description:
            print(f"   Descrição: {self.description}")
        if self.tags:
            print(f"   Tags: {', '.join(self.tags)}")
        print(f"   Git commit: {self.git_commit}")
        print(f"   Python: {self.python_version}")
        
        print(f"\n🌐 Ambiente:")
        print(f"   Perfil: {self.env_profile}")
        print(f"   N° ambientes: {self.n_envs}")
        print(f"   Tipo: {self.vec_env_type}")
        print(f"   Seed: {self.seed}")
        if self.env_overrides:
            print(f"   Overrides: {self.env_overrides}")
        
        print(f"\n🎯 Hiperparâmetros:")
        print(f"   Perfil: {self.hyperparam_profile}")
        if self.hyperparam_overrides:
            print(f"   Overrides: {self.hyperparam_overrides}")
        
        print(f"\n🏃 Treinamento:")
        print(f"   Total timesteps: {self.total_timesteps:,}")
        print(f"   Checkpoint freq: {self.checkpoint_freq:,}")
        print(f"   Log freq: {self.log_freq:,}")
        
        print(f"\n💾 Salvamento:")
        print(f"   Base dir: {self.base_dir}")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        if self.use_tensorboard:
            print(f"   Tensorboard: {self.tensorboard_dir}")
        print(f"   Logs: {self.logs_dir}")
        
        print(f"\n📊 Callbacks ativos:")
        callbacks = []
        if self.enable_blocking_rate_callback:
            callbacks.append("BlockingRate")
        if self.enable_fragmentation_callback:
            callbacks.append("Fragmentation")
        if self.enable_reward_stats_callback:
            callbacks.append("RewardStats")
        callbacks.append("Checkpoint")
        callbacks.append("TrainingMonitor")
        print(f"   {', '.join(callbacks)}")
        
        if self.notes:
            print(f"\n📝 Notas:")
            print(f"   {self.notes}")
        
        print("\n" + "="*80 + "\n")


def create_training_setup(
    config: TrainingConfig
) -> Tuple[Any, MaskablePPO, CallbackList]:
    """
    Cria e configura TUDO para treinamento baseado em TrainingConfig.
    
    Esta função é o ponto central que:
    1. Cria diretórios
    2. Salva configuração
    3. Cria ambiente vetorizado
    4. Cria modelo PPO
    5. Configura todos os callbacks
    
    Args:
        config: Objeto TrainingConfig com todas as configurações
        
    Returns:
        Tupla (env, model, callbacks) prontos para usar
        
    Example:
        >>> config = TrainingConfig(experiment_name="test")
        >>> env, model, callbacks = create_training_setup(config)
        >>> model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
    """
    
    if config.verbose >= 1:
        config.print_summary()
    
    # 1. Criar diretórios
    config.create_directories()
    
    # 2. Salvar configuração
    if config.save_config_to_file:
        config.save_to_file()
        config.save_to_json()
    
    # 3. Criar ambiente
    if config.verbose >= 1:
        print("\n🔨 Criando ambientes...")
    
    env_profile_map = {
        'fast': PROFILE_FAST,
        'default': PROFILE_DEFAULT,
        'intensive': PROFILE_INTENSIVE,
        'light_load': PROFILE_LIGHT_LOAD,
        'heavy_load': PROFILE_HEAVY_LOAD
    }
    env_profile = env_profile_map[config.env_profile]
    
    vec_env = create_vec_env(
        n_envs=config.n_envs,
        profile=env_profile,
        vec_env_cls=config.vec_env_type,
        seed=config.seed,
        **config.env_overrides
    )
    
    if config.verbose >= 1:
        print(f"✓ {config.n_envs} ambientes criados")
    
    # 4. Configurar hiperparâmetros
    if config.verbose >= 1:
        print("\n🎯 Configurando hiperparâmetros...")
    
    hyperparams = get_hyperparams(config.hyperparam_profile)
    
    # Aplicar overrides
    for key, value in config.hyperparam_overrides.items():
        if hasattr(hyperparams, key):
            setattr(hyperparams, key, value)
        else:
            warnings.warn(f"Unknown hyperparameter: {key}")
    
    # Configurar tensorboard
    if config.use_tensorboard:
        hyperparams.tensorboard_log = config.tensorboard_dir
    else:
        hyperparams.tensorboard_log = None
    
    hyperparams.verbose = config.verbose
    
    if config.verbose >= 1:
        print(f"✓ Hiperparâmetros configurados (perfil: {config.hyperparam_profile})")
    
    # 5. Criar modelo
    if config.verbose >= 1:
        print("\n🤖 Criando modelo MaskablePPO...")
    
    # Detectar tipo de policy baseado no observation space
    from gymnasium import spaces
    if isinstance(vec_env.observation_space, spaces.Dict):
        policy_type = "MultiInputPolicy"
    else:
        policy_type = "MlpPolicy"
    
    if config.verbose >= 2:
        print(f"   Policy type: {policy_type}")
    
    model = MaskablePPO(
        policy_type,
        vec_env,
        **hyperparams.to_dict()
    )
    
    if config.verbose >= 1:
        print("✓ Modelo criado")
        # Imprime info da rede
        if hasattr(model.policy, 'mlp_extractor'):
            print(f"   Policy network: {hyperparams.policy_kwargs.get('net_arch', {}).get('pi', 'N/A')}")
            print(f"   Value network: {hyperparams.policy_kwargs.get('net_arch', {}).get('vf', 'N/A')}")
    
    # 6. Criar callbacks
    if config.verbose >= 1:
        print("\n📊 Configurando callbacks...")
    
    callbacks_list = []
    
    # Training Monitor (sempre ativo)
    callbacks_list.append(
        TrainingMonitorCallback(
            verbose=config.verbose,
            log_freq=config.log_freq * 10
        )
    )
    
    # Blocking Rate
    if config.enable_blocking_rate_callback:
        callbacks_list.append(
            BlockingRateCallback(
                verbose=config.verbose,
                log_freq=config.log_freq
            )
        )
    
    # Reward Stats
    if config.enable_reward_stats_callback:
        callbacks_list.append(
            RewardStatsCallback(
                verbose=config.verbose,
                log_freq=config.log_freq
            )
        )
    
    # Fragmentation
    if config.enable_fragmentation_callback:
        callbacks_list.append(
            FragmentationCallback(
                verbose=config.verbose,
                log_freq=config.log_freq * 5
            )
        )
    
    # Checkpoint (sempre ativo)
    callbacks_list.append(
        CustomCheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=config.checkpoint_dir,
            name_prefix="ppo_qrmsa",
            save_best=config.save_best_model,
            verbose=config.verbose
        )
    )
    
    callbacks = CallbackList(callbacks_list)
    
    if config.verbose >= 1:
        print(f"✓ {len(callbacks_list)} callbacks configurados")
    
    # 7. Sumário final
    if config.verbose >= 1:
        print("\n" + "="*80)
        print(" SETUP COMPLETO - PRONTO PARA TREINAR")
        print("="*80)
        print("\n💡 Para iniciar o treinamento:")
        print("   model.learn(total_timesteps=config.total_timesteps, callback=callbacks)")
        if config.use_tensorboard:
            print(f"\n📊 Para monitorar no TensorBoard:")
            print(f"   tensorboard --logdir {config.tensorboard_dir}")
        print("\n" + "="*80 + "\n")
    
    return vec_env, model, callbacks


# ============================================================================
# CONFIGURAÇÕES PRÉ-DEFINIDAS PARA CASOS COMUNS
# ============================================================================

def create_quick_test_config() -> TrainingConfig:
    """
    Configuração para teste rápido (5-10 minutos).
    Útil para verificar se tudo está funcionando.
    """
    return TrainingConfig(
        experiment_name="quick_test",
        description="Teste rápido para validação",
        tags=["test", "quick"],
        env_profile="fast",
        hyperparam_profile="fast",
        total_timesteps=50000,
        n_envs=4,
        vec_env_type="dummy",
        checkpoint_freq=10000,
        log_freq=500,
        verbose=1
    )


def create_baseline_config() -> TrainingConfig:
    """
    Configuração baseline para estabelecer performance mínima.
    Treinamento de 2-3 horas.
    """
    return TrainingConfig(
        experiment_name="baseline",
        description="Baseline com configurações padrão",
        tags=["baseline", "default"],
        env_profile="default",
        hyperparam_profile="default",
        total_timesteps=500000,
        n_envs=8,
        vec_env_type="subproc",
        checkpoint_freq=50000,
        log_freq=1000,
        verbose=1
    )


def create_production_config() -> TrainingConfig:
    """
    Configuração para treinamento production (12-24 horas).
    Usa configurações intensivas e todos os callbacks.
    """
    return TrainingConfig(
        experiment_name="production_run",
        description="Treinamento production completo",
        tags=["production", "intensive", "full"],
        env_profile="intensive",
        hyperparam_profile="intensive",
        total_timesteps=5000000,
        n_envs=16,
        vec_env_type="subproc",
        checkpoint_freq=100000,
        log_freq=2000,
        save_best_model=True,
        enable_blocking_rate_callback=True,
        enable_fragmentation_callback=True,
        enable_reward_stats_callback=True,
        verbose=1
    )


if __name__ == "__main__":
    """
    Demonstração do sistema de configuração.
    """
    print("\n" + "="*80)
    print(" SISTEMA DE CONFIGURAÇÃO DE TREINAMENTO")
    print("="*80)
    
    print("\n📚 Configurações pré-definidas disponíveis:\n")
    
    configs = {
        "Quick Test": create_quick_test_config(),
        "Baseline": create_baseline_config(),
        "Production": create_production_config()
    }
    
    for name, config in configs.items():
        print(f"\n{'='*80}")
        print(f" {name.upper()}")
        print(f"{'='*80}")
        print(f"Timesteps: {config.total_timesteps:,}")
        print(f"Env profile: {config.env_profile}")
        print(f"Hyperparam profile: {config.hyperparam_profile}")
        print(f"N envs: {config.n_envs}")
        print(f"Description: {config.description}")
    
    print("\n" + "="*80)
    print("\n💡 Exemplo de uso:")
    print("""
from training_config import TrainingConfig, create_training_setup

# Opção 1: Usar config pré-definida
from training_config import create_baseline_config
config = create_baseline_config()

# Opção 2: Criar config personalizada
config = TrainingConfig(
    experiment_name="meu_experimento",
    description="Testando nova arquitetura",
    env_profile="default",
    hyperparam_profile="default",
    total_timesteps=1000000,
    n_envs=8
)

# Setup completo
env, model, callbacks = create_training_setup(config)

# Treinar
model.learn(total_timesteps=config.total_timesteps, callback=callbacks)

# Salvar modelo final
model.save(f"{config.checkpoint_dir}/final_model")
    """)
    print("="*80 + "\n")
