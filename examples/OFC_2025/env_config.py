#!/usr/bin/env python3
"""
Configuração Otimizada do Ambiente QRMSA para Treinamento com PPO

Este módulo centraliza todas as configurações do ambiente, permitindo:
- Criação consistente de ambientes para treinamento e teste
- Fácil ajuste de parâmetros experimentais
- Diferentes perfis de configuração (rápido, padrão, completo)
- Suporte para múltiplas topologias

Autor: Criado para OFC 2025
Data: Outubro 2025
"""

import os
import random
from typing import Tuple, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import numpy as np
import gymnasium as gym

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from action_mask_wrapper import ActionMaskWrapper


# =========================================================
# DEFINIÇÕES DE MODULAÇÕES
# =========================================================

def get_modulations_standard() -> Tuple[Modulation, ...]:
    """
    Conjunto padrão de 6 modulações (BPSK a 64QAM).
    
    Baseado em parâmetros realistas de sistemas ópticos coerentes.
    Recomendado para experimentos completos.
    """
    return (
        Modulation(
            name="BPSK",
            maximum_length=100000,  # Praticamente ilimitado
            spectral_efficiency=1,
            minimum_osnr=3.71925646843142,
            inband_xt=-14
        ),
        Modulation(
            name="QPSK",
            maximum_length=2000,
            spectral_efficiency=2,
            minimum_osnr=6.72955642507124,
            inband_xt=-17
        ),
        Modulation(
            name="8QAM",
            maximum_length=1000,
            spectral_efficiency=3,
            minimum_osnr=10.8453935345953,
            inband_xt=-20
        ),
        Modulation(
            name="16QAM",
            maximum_length=500,
            spectral_efficiency=4,
            minimum_osnr=13.2406469649752,
            inband_xt=-23
        ),
        Modulation(
            name="32QAM",
            maximum_length=250,
            spectral_efficiency=5,
            minimum_osnr=16.1608982942870,
            inband_xt=-26
        ),
        Modulation(
            name="64QAM",
            maximum_length=125,
            spectral_efficiency=6,
            minimum_osnr=19.0134649345090,
            inband_xt=-29
        ),
    )


def get_modulations_simplified() -> Tuple[Modulation, ...]:
    """
    Conjunto simplificado de 2 modulações (BPSK e QPSK).
    
    Recomendado para:
    - Testes rápidos e debugging
    - Prototipagem de algoritmos
    - Ambientes com espaço de ações reduzido
    """
    return (
        Modulation(
            name="BPSK",
            maximum_length=100000,
            spectral_efficiency=1,
            minimum_osnr=3.71925646843142,
            inband_xt=-14
        ),
        Modulation(
            name="QPSK",
            maximum_length=2000,
            spectral_efficiency=2,
            minimum_osnr=6.72955642507124,
            inband_xt=-17
        ),
    )


def get_modulations_extended() -> Tuple[Modulation, ...]:
    """
    Conjunto estendido com modulações adicionais.
    
    Inclui variantes intermediárias para maior granularidade.
    """
    # Por enquanto retorna o padrão, pode ser estendido no futuro
    return get_modulations_standard()


# =========================================================
# CONFIGURAÇÕES DE TOPOLOGIA
# =========================================================

@dataclass
class TopologyConfig:
    """Configuração de topologia de rede."""
    
    name: str
    file_path: str
    max_span_length: float = 80.0  # km
    default_attenuation: float = 0.2  # dB/km
    default_noise_figure: float = 4.5  # dB
    k_paths: int = 5
    
    def __post_init__(self):
        """Valida se o arquivo existe."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Topology file not found: {self.file_path}")


def get_topology_config(
    topology_name: Literal["nobel-eu", "germany50", "nsfnet", "custom"] = "nobel-eu",
    custom_path: Optional[str] = None,
    base_dir: Optional[str] = None
) -> TopologyConfig:
    """
    Retorna a configuração de topologia.
    
    Args:
        topology_name: Nome da topologia predefinida
        custom_path: Caminho customizado (se topology_name="custom")
        base_dir: Diretório base onde procurar topologias
        
    Returns:
        TopologyConfig com os parâmetros da topologia
    """
    if base_dir is None:
        # Assume que está em examples/OFC_2025/
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "topologies"
        )
    
    topology_files = {
        "nobel-eu": "nobel-eu.xml",
        "germany50": "germany50.xml",
        "nsfnet": "nsfnet_chen.txt",
    }
    
    if topology_name == "custom":
        if custom_path is None:
            raise ValueError("custom_path must be provided when topology_name='custom'")
        return TopologyConfig(
            name=os.path.splitext(os.path.basename(custom_path))[0],
            file_path=custom_path
        )
    
    if topology_name not in topology_files:
        raise ValueError(
            f"Unknown topology: {topology_name}. "
            f"Available: {list(topology_files.keys())}"
        )
    
    file_path = os.path.join(base_dir, topology_files[topology_name])
    
    return TopologyConfig(
        name=topology_name,
        file_path=file_path
    )


# =========================================================
# PERFIS DE CONFIGURAÇÃO DO AMBIENTE
# =========================================================

@dataclass
class EnvironmentProfile:
    """
    Perfil completo de configuração do ambiente.
    
    Agrupa todos os parâmetros necessários para criar um ambiente QRMSA.
    """
    
    # Identificação
    name: str = "default"
    description: str = ""
    
    # Topologia
    topology_name: str = "nobel-eu"
    modulations_preset: Literal["standard", "simplified", "extended"] = "standard"
    modulations_to_consider: int = 6  # Quantas modulações o agente pode escolher
    k_paths: int = 5
    
    # Recursos espectrais
    num_spectrum_resources: int = 320
    frequency_slot_bandwidth: float = 12.5e9  # Hz
    frequency_start: float = 3e8 / 1565e-9  # Hz (banda C)
    
    # Demanda de tráfego
    load: float = 300  # Erlang
    bit_rate_selection: Literal["discrete", "continuous"] = "discrete"
    bit_rates: Tuple[int, ...] = (10, 40, 100, 400)  # Gbps
    
    # Parâmetros físicos
    launch_power_dbm: float = 1.0  # dBm
    margin: float = 0.0  # dB (margem de OSNR)
    
    # Episódio
    episode_length: int = 1000  # Número de chegadas de serviço
    
    # Funcionalidades
    allow_rejection: bool = True
    measure_disruptions: bool = False
    defragmentation: bool = False
    n_defrag_services: int = 0
    gen_observation: bool = True
    rl_mode: bool = True  # Não lança exceções
    
    # Reprodutibilidade
    seed: Optional[int] = None
    
    # Logging
    file_name: str = ""  # Caminho para salvar estatísticas (vazio = não salva)
    
    def __post_init__(self):
        """Validações pós-inicialização."""
        # Calcula bandwidth automaticamente
        self.bandwidth = self.num_spectrum_resources * self.frequency_slot_bandwidth
        
        # Ajusta modulations_to_consider baseado no preset
        if self.modulations_preset == "simplified" and self.modulations_to_consider > 2:
            print(f"⚠ Warning: modulations_to_consider={self.modulations_to_consider} "
                  f"but preset='simplified' has only 2 modulations. Adjusting to 2.")
            self.modulations_to_consider = 2


# Perfis predefinidos
PROFILE_FAST = EnvironmentProfile(
    name="fast",
    description="Configuração rápida para testes e debugging",
    modulations_preset="simplified",
    modulations_to_consider=2,
    episode_length=100,
    load=200,
    num_spectrum_resources=160,
)

PROFILE_DEFAULT = EnvironmentProfile(
    name="default",
    description="Configuração padrão balanceada para treinamento",
    modulations_preset="standard",
    modulations_to_consider=6,
    episode_length=1000,
    load=300,
    num_spectrum_resources=320,
)

PROFILE_INTENSIVE = EnvironmentProfile(
    name="intensive",
    description="Configuração intensiva para experimentos completos",
    modulations_preset="standard",
    modulations_to_consider=6,
    episode_length=2000,
    load=400,
    num_spectrum_resources=320,
    measure_disruptions=True,
)

PROFILE_LIGHT_LOAD = EnvironmentProfile(
    name="light_load",
    description="Carga leve para análise de convergência",
    modulations_preset="standard",
    modulations_to_consider=6,
    episode_length=1000,
    load=150,
    num_spectrum_resources=320,
)

PROFILE_HEAVY_LOAD = EnvironmentProfile(
    name="heavy_load",
    description="Carga pesada para testar limites",
    modulations_preset="standard",
    modulations_to_consider=6,
    episode_length=1000,
    load=500,
    num_spectrum_resources=320,
)


# =========================================================
# FUNÇÃO PRINCIPAL DE CRIAÇÃO DE AMBIENTES
# =========================================================

def create_qrmsa_env(
    profile: EnvironmentProfile = PROFILE_DEFAULT,
    topology_config: Optional[TopologyConfig] = None,
    apply_mask_wrapper: bool = True,
    seed: Optional[int] = None,
    rank: int = 0,  # Para ambientes paralelos
    **override_params
) -> gym.Env:
    """
    Cria um ambiente QRMSA configurado e pronto para uso.
    
    Args:
        profile: Perfil de configuração a usar
        topology_config: Configuração de topologia (None = usa padrão do profile)
        apply_mask_wrapper: Se True, aplica ActionMaskWrapper
        seed: Seed para reprodutibilidade (None = usa do profile)
        rank: Rank do ambiente (para paralelização)
        **override_params: Parâmetros adicionais para sobrescrever o profile
        
    Returns:
        env: Ambiente Gymnasium configurado
        
    Example:
        >>> # Criar ambiente padrão
        >>> env = create_qrmsa_env()
        
        >>> # Criar ambiente rápido para testes
        >>> env = create_qrmsa_env(profile=PROFILE_FAST, seed=42)
        
        >>> # Criar com overrides
        >>> env = create_qrmsa_env(
        ...     profile=PROFILE_DEFAULT,
        ...     load=350,
        ...     episode_length=1500
        ... )
    """
    # Configuração de seed
    if seed is None:
        seed = profile.seed if profile.seed is not None else 42
    
    # Adiciona rank ao seed para ambientes paralelos
    env_seed = seed + rank
    
    # Define seeds globais
    random.seed(env_seed)
    np.random.seed(env_seed)
    
    # Configuração de topologia
    if topology_config is None:
        topology_config = get_topology_config(profile.topology_name)
    
    # Seleciona modulações
    modulations_map = {
        "standard": get_modulations_standard,
        "simplified": get_modulations_simplified,
        "extended": get_modulations_extended,
    }
    modulations = modulations_map[profile.modulations_preset]()
    
    # Carrega topologia
    topology = get_topology(
        topology_config.file_path,
        topology_config.name,
        modulations,
        topology_config.max_span_length,
        topology_config.default_attenuation,
        topology_config.default_noise_figure,
        topology_config.k_paths
    )
    
    # Monta argumentos do ambiente
    env_args = dict(
        topology=topology,
        seed=env_seed,
        allow_rejection=profile.allow_rejection,
        load=profile.load,
        episode_length=profile.episode_length,
        num_spectrum_resources=profile.num_spectrum_resources,
        launch_power_dbm=profile.launch_power_dbm,
        bandwidth=profile.bandwidth,
        frequency_start=profile.frequency_start,
        frequency_slot_bandwidth=profile.frequency_slot_bandwidth,
        bit_rate_selection=profile.bit_rate_selection,
        bit_rates=profile.bit_rates,
        margin=profile.margin,
        file_name=profile.file_name,
        measure_disruptions=profile.measure_disruptions,
        k_paths=profile.k_paths,
        modulations_to_consider=profile.modulations_to_consider,
        defragmentation=profile.defragmentation,
        n_defrag_services=profile.n_defrag_services,
        gen_observation=profile.gen_observation,
        rl_mode=profile.rl_mode,
    )
    
    # Aplica overrides
    env_args.update(override_params)
    
    # Cria ambiente diretamente (sem usar gym.make)
    env = QRMSAEnvWrapper(**env_args)
    
    # Aplica wrapper de máscaras se solicitado
    if apply_mask_wrapper:
        env = ActionMaskWrapper(env)
    
    return env


def create_vec_env(
    n_envs: int = 4,
    profile: EnvironmentProfile = PROFILE_DEFAULT,
    vec_env_cls: Literal["dummy", "subproc"] = "subproc",
    seed: int = 42,
    **override_params
) -> gym.vector.VectorEnv:
    """
    Cria um ambiente vetorizado para treinamento paralelo.
    
    Args:
        n_envs: Número de ambientes paralelos
        profile: Perfil de configuração
        vec_env_cls: Tipo de vetorização ("dummy" ou "subproc")
        seed: Seed base (cada env terá seed + rank)
        **override_params: Parâmetros para sobrescrever
        
    Returns:
        vec_env: Ambiente vetorizado
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    
    def make_env(rank):
        """Função factory para criar ambiente."""
        def _init():
            return create_qrmsa_env(
                profile=profile,
                seed=seed,
                rank=rank,
                **override_params
            )
        return _init
    
    # Cria lista de funções factory
    env_fns = [make_env(i) for i in range(n_envs)]
    
    # Escolhe classe de vetorização
    if vec_env_cls == "dummy":
        vec_env = DummyVecEnv(env_fns)
    elif vec_env_cls == "subproc":
        vec_env = SubprocVecEnv(env_fns)
    else:
        raise ValueError(f"Unknown vec_env_cls: {vec_env_cls}")
    
    return vec_env


# =========================================================
# UTILITÁRIOS
# =========================================================

def print_env_info(env: gym.Env, profile: Optional[EnvironmentProfile] = None):
    """
    Imprime informações detalhadas sobre o ambiente.
    
    Args:
        env: Ambiente a ser analisado
        profile: Perfil usado (opcional, para contexto)
    """
    print("\n" + "="*70)
    print(" INFORMAÇÕES DO AMBIENTE ")
    print("="*70)
    
    if profile:
        print(f"\n📋 Perfil: {profile.name}")
        print(f"   {profile.description}")
    
    print(f"\n🎮 Espaços:")
    print(f"   Action Space: {env.action_space}")
    print(f"   Observation Space: {env.observation_space}")
    
    # Acessa o ambiente base (remove wrappers)
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    if hasattr(base_env, 'topology'):
        print(f"\n🌐 Topologia:")
        print(f"   Nome: {base_env.topology.name if hasattr(base_env.topology, 'name') else 'N/A'}")
        print(f"   Nós: {base_env.topology.number_of_nodes()}")
        print(f"   Enlaces: {base_env.topology.number_of_edges()}")
        print(f"   K-paths: {base_env.k_paths}")
    
    if hasattr(base_env, 'modulations'):
        print(f"\n📡 Modulações:")
        for mod in base_env.modulations:
            print(f"   - {mod.name}: SE={mod.spectral_efficiency}, "
                  f"OSNR_min={mod.minimum_osnr:.2f}dB")
    
    if hasattr(base_env, 'num_spectrum_resources'):
        print(f"\n📊 Recursos:")
        print(f"   Slots espectrais: {base_env.num_spectrum_resources}")
        if hasattr(base_env, 'bandwidth'):
            print(f"   Bandwidth: {base_env.bandwidth/1e9:.1f} GHz")
        if hasattr(base_env, 'frequency_slot_bandwidth'):
            print(f"   Slot bandwidth: {base_env.frequency_slot_bandwidth/1e9:.1f} GHz")
    
    if hasattr(base_env, 'load'):
        print(f"\n🚦 Tráfego:")
        print(f"   Carga: {base_env.load} Erlang")
        print(f"   Bit rates: {base_env.bit_rates}")
        if hasattr(base_env, 'episode_length'):
            print(f"   Episode length: {base_env.episode_length}")
    
    print("\n" + "="*70)


def get_available_profiles() -> Dict[str, EnvironmentProfile]:
    """
    Retorna dicionário com todos os perfis disponíveis.
    
    Returns:
        profiles: Dict[nome, perfil]
    """
    return {
        "fast": PROFILE_FAST,
        "default": PROFILE_DEFAULT,
        "intensive": PROFILE_INTENSIVE,
        "light_load": PROFILE_LIGHT_LOAD,
        "heavy_load": PROFILE_HEAVY_LOAD,
    }
