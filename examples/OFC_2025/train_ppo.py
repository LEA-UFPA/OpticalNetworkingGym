#!/usr/bin/env python3
"""
Script de treinamento PPO para QRMSA Environment

Este script integra todos os componentes para treinamento completo:
- Configuração de ambiente (env_config.py)
- Hiperparâmetros otimizados (hyperparams.py)
- Callbacks de monitoramento (callbacks.py)
- MaskablePPO com action masking

Uso:
    # Treinamento rápido (debugging)
    python train_ppo.py --env-profile fast --hyperparam-profile fast --timesteps 100000
    
    # Treinamento padrão
    python train_ppo.py --env-profile default --hyperparam-profile default --timesteps 1000000
    
    # Treinamento intensivo
    python train_ppo.py --env-profile intensive --hyperparam-profile intensive --timesteps 5000000
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

# Adiciona path do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importa módulos locais
from env_config import (
    create_vec_env,
    get_available_profiles as get_env_profiles,
    PROFILE_FAST,
    PROFILE_DEFAULT,
    PROFILE_INTENSIVE,
    PROFILE_LIGHT_LOAD,
    PROFILE_HEAVY_LOAD
)
from hyperparams import (
    get_hyperparams,
    print_hyperparams,
    get_available_profiles as get_hyperparam_profiles
)
from callbacks import create_callbacks


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treinamento PPO para QRMSA Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Teste rápido (100k steps, ~5-10 min)
  python train_ppo.py --env-profile fast --hyperparam-profile fast --timesteps 100000

  # Treinamento padrão (1M steps, ~2-3 horas)
  python train_ppo.py --env-profile default --hyperparam-profile default --timesteps 1000000

  # Treinamento completo (5M steps, ~10-15 horas)
  python train_ppo.py --env-profile intensive --hyperparam-profile intensive --timesteps 5000000

  # Continuar treinamento de checkpoint
  python train_ppo.py --load-path ./models/ppo_qrmsa_1000000.zip --timesteps 1000000

  # Ambiente personalizado
  python train_ppo.py --env-profile default --load 350 --episode-length 1500 --timesteps 2000000
        """
    )
    
    # Ambiente
    env_group = parser.add_argument_group('Configuração do Ambiente')
    env_group.add_argument(
        '--env-profile',
        type=str,
        default='default',
        choices=['fast', 'default', 'intensive', 'light_load', 'heavy_load'],
        help='Perfil de configuração do ambiente'
    )
    env_group.add_argument(
        '--n-envs',
        type=int,
        default=8,
        help='Número de ambientes paralelos (default: 8)'
    )
    env_group.add_argument(
        '--vec-env-type',
        type=str,
        default='subproc',
        choices=['dummy', 'subproc'],
        help='Tipo de ambiente vetorizado (default: subproc)'
    )
    env_group.add_argument(
        '--load',
        type=float,
        default=None,
        help='Override: carga de tráfego em Erlang'
    )
    env_group.add_argument(
        '--episode-length',
        type=int,
        default=None,
        help='Override: número de eventos por episódio'
    )
    
    # Hiperparâmetros
    hyper_group = parser.add_argument_group('Hiperparâmetros')
    hyper_group.add_argument(
        '--hyperparam-profile',
        type=str,
        default='default',
        choices=['fast', 'default', 'intensive', 'high_exploration', 'stable'],
        help='Perfil de hiperparâmetros'
    )
    hyper_group.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override: taxa de aprendizado'
    )
    
    # Treinamento
    train_group = parser.add_argument_group('Treinamento')
    train_group.add_argument(
        '--timesteps',
        type=int,
        default=1000000,
        help='Número total de timesteps (default: 1000000)'
    )
    train_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para reprodutibilidade (default: 42)'
    )
    train_group.add_argument(
        '--load-path',
        type=str,
        default=None,
        help='Caminho para carregar modelo existente'
    )
    
    # Salvamento e logging
    save_group = parser.add_argument_group('Salvamento e Logging')
    save_group.add_argument(
        '--save-dir',
        type=str,
        default='./models',
        help='Diretório para salvar modelos (default: ./models)'
    )
    save_group.add_argument(
        '--tensorboard-log',
        type=str,
        default='./tensorboard_logs',
        help='Diretório para logs do TensorBoard (default: ./tensorboard_logs)'
    )
    save_group.add_argument(
        '--checkpoint-freq',
        type=int,
        default=50000,
        help='Frequência de checkpoints em steps (default: 50000)'
    )
    save_group.add_argument(
        '--log-freq',
        type=int,
        default=1000,
        help='Frequência de logging em steps (default: 1000)'
    )
    save_group.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Nome da execução (default: timestamp automático)'
    )
    
    # Outros
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Nível de verbosidade (0: quiet, 1: info, 2: debug)'
    )
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Desabilita logging no TensorBoard'
    )
    
    return parser.parse_args()


def setup_directories(args):
    """Cria diretórios necessários."""
    os.makedirs(args.save_dir, exist_ok=True)
    if not args.no_tensorboard:
        os.makedirs(args.tensorboard_log, exist_ok=True)
    
    return True


def create_run_name(args):
    """Cria nome único para a execução."""
    if args.run_name:
        return args.run_name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{args.env_profile}_{args.hyperparam_profile}_{timestamp}"
    
    return run_name


def print_training_config(args, run_name, hyperparams, env_profile):
    """Imprime configuração do treinamento."""
    print("\n" + "="*80)
    print(" CONFIGURAÇÃO DO TREINAMENTO PPO")
    print("="*80)
    
    print(f"\n🏷️  Run: {run_name}")
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌱 Seed: {args.seed}")
    
    print(f"\n🌐 Ambiente:")
    print(f"   Perfil: {args.env_profile}")
    print(f"   Load: {env_profile.load} Erlang")
    print(f"   Episode length: {env_profile.episode_length}")
    print(f"   Spectrum slots: {env_profile.num_spectrum_resources}")
    print(f"   Modulações: {env_profile.modulations_type} ({len(env_profile.get_modulations())})")
    print(f"   Ambientes paralelos: {args.n_envs}")
    print(f"   Tipo: {args.vec_env_type}")
    
    print(f"\n🎯 Hiperparâmetros:")
    print(f"   Perfil: {args.hyperparam_profile}")
    lr = hyperparams.learning_rate
    lr_str = f"{lr:.2e}" if not callable(lr) else "linear_schedule"
    print(f"   Learning rate: {lr_str}")
    print(f"   N steps: {hyperparams.n_steps}")
    print(f"   Batch size: {hyperparams.batch_size}")
    print(f"   N epochs: {hyperparams.n_epochs}")
    
    print(f"\n🏃 Treinamento:")
    print(f"   Total timesteps: {args.timesteps:,}")
    estimated_episodes = args.timesteps / env_profile.episode_length
    print(f"   Episódios estimados: ~{estimated_episodes:,.0f}")
    print(f"   Updates estimados: ~{args.timesteps // hyperparams.n_steps:,}")
    
    print(f"\n💾 Salvamento:")
    print(f"   Diretório: {args.save_dir}")
    print(f"   Checkpoint freq: {args.checkpoint_freq:,} steps")
    if not args.no_tensorboard:
        print(f"   TensorBoard: {args.tensorboard_log}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Função principal."""
    args = parse_args()
    
    # Setup
    setup_directories(args)
    run_name = create_run_name(args)
    
    # Mapeamento de perfis de ambiente
    env_profile_map = {
        'fast': PROFILE_FAST,
        'default': PROFILE_DEFAULT,
        'intensive': PROFILE_INTENSIVE,
        'light_load': PROFILE_LIGHT_LOAD,
        'heavy_load': PROFILE_HEAVY_LOAD
    }
    env_profile = env_profile_map[args.env_profile]
    
    # Aplica overrides se fornecidos
    env_kwargs = {}
    if args.load is not None:
        env_kwargs['load'] = args.load
    if args.episode_length is not None:
        env_kwargs['episode_length'] = args.episode_length
    
    # Obtém hiperparâmetros
    hyperparams = get_hyperparams(args.hyperparam_profile)
    
    # Aplica overrides de hiperparâmetros
    if args.learning_rate is not None:
        hyperparams.learning_rate = args.learning_rate
    
    # Configura tensorboard
    if args.no_tensorboard:
        hyperparams.tensorboard_log = None
    else:
        hyperparams.tensorboard_log = os.path.join(args.tensorboard_log, run_name)
    
    # Imprime configuração
    print_training_config(args, run_name, hyperparams, env_profile)
    
    # Cria ambiente vetorizado
    print("🔨 Criando ambientes...")
    vec_env = create_vec_env(
        n_envs=args.n_envs,
        profile=env_profile,
        vec_env_cls=args.vec_env_type,
        seed=args.seed,
        **env_kwargs
    )
    print(f"✓ {args.n_envs} ambientes criados")
    
    # Cria ou carrega modelo
    if args.load_path and os.path.exists(args.load_path):
        print(f"\n📂 Carregando modelo de: {args.load_path}")
        model = MaskablePPO.load(
            args.load_path,
            env=vec_env,
            verbose=args.verbose
        )
        print("✓ Modelo carregado")
    else:
        print("\n🤖 Criando novo modelo MaskablePPO...")
        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
            **hyperparams.to_dict(),
            verbose=args.verbose
        )
        print("✓ Modelo criado")
    
    # Cria callbacks
    print("\n📊 Configurando callbacks...")
    callbacks = create_callbacks(
        save_path=os.path.join(args.save_dir, run_name),
        checkpoint_freq=args.checkpoint_freq,
        log_freq=args.log_freq,
        verbose=args.verbose
    )
    print(f"✓ {len(callbacks)} callbacks configurados")
    
    # Informações finais antes de iniciar
    print("\n" + "="*80)
    print(" INICIANDO TREINAMENTO")
    print("="*80)
    print("\n💡 Dicas:")
    print("   - Use Ctrl+C para interromper graciosamente")
    if not args.no_tensorboard:
        print(f"   - Monitore com: tensorboard --logdir {args.tensorboard_log}")
    print(f"   - Checkpoints salvos em: {os.path.join(args.save_dir, run_name)}")
    print("\n" + "="*80 + "\n")
    
    # Inicia treinamento
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Salva modelo final
        final_path = os.path.join(args.save_dir, run_name, f"final_model_{args.timesteps}")
        model.save(final_path)
        
        print("\n" + "="*80)
        print(" TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"\n✓ Modelo final salvo em: {final_path}.zip")
        print(f"✓ Total timesteps: {args.timesteps:,}")
        print(f"✓ Run name: {run_name}")
        print("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print(" TREINAMENTO INTERROMPIDO PELO USUÁRIO")
        print("="*80)
        
        # Salva modelo atual
        interrupted_path = os.path.join(args.save_dir, run_name, f"interrupted_{model.num_timesteps}")
        model.save(interrupted_path)
        
        print(f"\n✓ Modelo salvo em: {interrupted_path}.zip")
        print(f"✓ Timesteps completados: {model.num_timesteps:,}")
        print("\n" + "="*80 + "\n")
    
    finally:
        # Limpa ambiente
        vec_env.close()
        print("✓ Ambientes fechados")


if __name__ == "__main__":
    main()
