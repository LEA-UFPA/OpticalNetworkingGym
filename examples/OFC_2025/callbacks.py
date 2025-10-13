#!/usr/bin/env python3
"""
Callbacks customizados para monitoramento de treinamento PPO

Este módulo implementa callbacks especializados para monitorar métricas
específicas do ambiente QRMSA durante o treinamento com MaskablePPO.

Callbacks disponíveis:
- BlockingRateCallback: Monitora taxa de bloqueio
- FragmentationCallback: Analisa fragmentação espectral
- RewardStatsCallback: Estatísticas detalhadas de rewards
- CustomCheckpointCallback: Salva melhores modelos
- TrainingMonitorCallback: Combina múltiplas métricas
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

try:
    from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
except ImportError:  # pragma: no cover
    TorchSummaryWriter = None

try:
    from tensorboardX import SummaryWriter as TensorboardXSummaryWriter  # type: ignore
except ImportError:  # pragma: no cover
    TensorboardXSummaryWriter = None


class BlockingRateCallback(BaseCallback):
    """Registra a taxa de bloqueio apenas quando um episódio termina."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])

        for done, info in zip(dones, infos):
            if not done or not isinstance(info, dict):
                continue

            bp = info.get('episode_service_blocking_rate')
            if bp is None:
                bp = info.get('service_blocking_rate')

            if bp is None:
                continue

            value = float(bp)
            self.logger.record("blocking_probability/episode", value)
            self.logger.record("blocking_probability/episode_percent", value * 100.0)

            if self.verbose >= 1:
                print(f"[BlockingRate] episódio com blocking={value * 100:.2f}%")

            self.logger.dump(step=self.num_timesteps)

        return True


class FragmentationCallback(BaseCallback):
    """Registra uso de modulação e métricas de defrag ao final de episódios."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.modulation_keys: List[str] | None = None
        self._tb_writer = None

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])

        for done, info in zip(dones, infos):
            if not done or not isinstance(info, dict):
                continue

            self._log_episode(info)

        return True

    def _log_episode(self, info: Dict[str, Any]) -> None:
        self._ensure_modulation_keys(info)

        mod_values: Dict[str, float] = {}
        if self.modulation_keys:
            for key in self.modulation_keys:
                value = info.get(key)
                if value is None:
                    continue
                label = self._format_modulation_label(key)
                mod_values[label] = float(value)

        if mod_values:
            self._log_modulation_usage(mod_values)

        defrag_metrics = {
            'episode_defrag_cicles': info.get('episode_defrag_cicles'),
            'episode_service_realocations': info.get('episode_service_realocations'),
        }

        frag_metrics = {
            'fragmentation_shannon_entropy': info.get('fragmentation_shannon_entropy'),
            'fragmentation_route_cuts': info.get('fragmentation_route_cuts'),
            'fragmentation_route_rss': info.get('fragmentation_route_rss'),
        }

        for key, value in defrag_metrics.items():
            if value is not None:
                self.logger.record(f"defrag/{key}", float(value))

        for key, value in frag_metrics.items():
            if value is not None:
                self.logger.record(f"fragmentation/{key}", float(value))

        if self.verbose >= 1 and mod_values:
            print(f"[Fragmentation] Uso de modulação: {mod_values}")

        self.logger.dump(step=self.num_timesteps)

    def _log_modulation_usage(self, values: Dict[str, float]) -> None:
        writer = self._get_tb_writer()
        if writer is not None:
            writer.add_scalars("modulation_usage", values, self.num_timesteps)
            writer.flush()
        else:
            for label, value in values.items():
                self.logger.record(f"modulation_usage/{label}", value)

    def _ensure_modulation_keys(self, info: Dict[str, Any]) -> None:
        if self.modulation_keys is not None:
            return
        mod_keys = [key for key in info.keys() if key.startswith('modulation_')]
        if mod_keys:
            self.modulation_keys = sorted(mod_keys)

    @staticmethod
    def _format_modulation_label(key: str) -> str:
        suffix = key.split('modulation_', 1)[-1]
        try:
            idx = int(float(suffix))
            return f"mod_{idx}"
        except ValueError:
            clean = suffix.replace('.', '_')
            return f"mod_{clean}"

    def _get_tb_writer(self):
        if self._tb_writer is not None:
            return self._tb_writer

        writer_cls = TorchSummaryWriter or TensorboardXSummaryWriter
        if writer_cls is None:
            return None

        log_dir = getattr(self.logger, "dir", None)
        if log_dir is None:
            return None

        custom_dir = Path(log_dir) / "modulation_usage"
        custom_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = writer_cls(log_dir=str(custom_dir))
        return self._tb_writer

    def _on_training_end(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None


class RewardStatsCallback(BaseCallback):
    """Registra reward por episódio e média por step."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.current_episode_rewards: list[list[float]] | None = None
        self.last_step_mean: float = 0.0

    def _init_callback(self) -> None:
        n_envs = self.training_env.num_envs if isinstance(self.training_env, VecEnv) else 1
        self.current_episode_rewards = [[] for _ in range(n_envs)]

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])

        rewards_array = np.array(rewards, dtype=float).reshape(-1) if len(rewards) > 0 else np.array([])
        if rewards_array.size > 0:
            self.last_step_mean = float(np.mean(rewards_array))
            for idx, reward in enumerate(rewards_array):
                self.current_episode_rewards[idx].append(float(reward))

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if not done or not isinstance(info, dict):
                continue

            episode_reward = self._extract_episode_reward(info, env_idx)

            self.logger.record("reward/episode", episode_reward)
            self.logger.record("reward/mean_step", self.last_step_mean)

            if self.verbose >= 1:
                print(f"[RewardStats] episódio reward={episode_reward:.3f} | mean_step={self.last_step_mean:.3f}")

            self.logger.dump(step=self.num_timesteps)
            self.current_episode_rewards[env_idx] = []

        return True

    def _extract_episode_reward(self, info: Dict[str, Any], env_idx: int) -> float:
        episode_reward = info.get('episode_reward')

        if episode_reward is None:
            episode_data = info.get('episode')
            if isinstance(episode_data, dict) and 'r' in episode_data:
                episode_reward = episode_data['r']

        if episode_reward is None and self.current_episode_rewards is not None:
            episode_reward = float(np.sum(self.current_episode_rewards[env_idx]))

        if episode_reward is None:
            episode_reward = 0.0

        return float(episode_reward)


class CustomCheckpointCallback(CheckpointCallback):
    """
    Checkpoint callback extendido que salva baseado em melhor performance.
    
    Além dos checkpoints periódicos, salva o melhor modelo baseado em:
        - Maior reward médio
        - Menor taxa de bloqueio
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
        save_best: bool = True
    ):
        """
        Args:
            save_freq: Frequência de salvamento (em steps)
            save_path: Diretório para salvar checkpoints
            name_prefix: Prefixo do nome dos arquivos
            save_replay_buffer: Se deve salvar replay buffer
            save_vecnormalize: Se deve salvar VecNormalize
            verbose: Nível de verbosidade
            save_best: Se deve salvar o melhor modelo
        """
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose
        )
        self.save_best = save_best
        self.best_mean_reward = -np.inf
        self.best_model_path = None
        
    def _on_step(self) -> bool:
        """Verifica se deve salvar checkpoint ou melhor modelo."""
        # Checkpoint periódico (comportamento padrão)
        result = super()._on_step()
        
        # Verifica se deve salvar melhor modelo
        if self.save_best and self.n_calls % self.save_freq == 0:
            self._check_and_save_best()
        
        return result
    
    def _check_and_save_best(self) -> None:
        """Salva modelo se for o melhor até agora."""
        # Tenta obter reward médio das últimas avaliações
        # Nota: isso requer que EvalCallback esteja sendo usado
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                # Salva melhor modelo
                best_path = os.path.join(self.save_path, f"{self.name_prefix}_best")
                self.model.save(best_path)
                self.best_model_path = best_path
                
                if self.verbose >= 1:
                    print(f"\n[Checkpoint] New best model saved! Mean reward: {mean_reward:.2f}")
                    print(f"             Saved to: {best_path}")


class TrainingMonitorCallback(BaseCallback):
    """
    Callback que combina múltiplas métricas de monitoramento.
    
    Este é um callback "meta" que agrega informações de outros callbacks
    e fornece uma visão geral do progresso do treinamento.
    """
    
    def __init__(self, verbose: int = 1, log_freq: int = 10000):
        """
        Args:
            verbose: Nível de verbosidade
            log_freq: Frequência de relatório completo
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.training_start_time = None
        
    def _on_training_start(self) -> None:
        """Marca início do treinamento."""
        import time
        self.training_start_time = time.time()
        
        if self.verbose >= 1:
            print("\n" + "="*70)
            print(" TREINAMENTO INICIADO")
            print("="*70)
            print(f"Total timesteps: {self.locals.get('total_timesteps', 'N/A')}")
            print(f"Number of environments: {self.training_env.num_envs}")
            print("="*70 + "\n")
    
    def _on_step(self) -> bool:
        """Monitora progresso."""
        if self.n_calls % self.log_freq == 0:
            self._print_progress_report()
        
        return True
    
    def _print_progress_report(self) -> None:
        """Imprime relatório de progresso."""
        if self.verbose < 1:
            return
        
        import time
        elapsed = time.time() - self.training_start_time
        
        print("\n" + "="*70)
        print(f" PROGRESSO DO TREINAMENTO - Step {self.n_calls}")
        print("="*70)
        
        # Tempo decorrido
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"⏱️  Tempo: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Steps por segundo
        steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
        print(f"🚀 Velocidade: {steps_per_sec:.1f} steps/sec")
        
        # Memória (se disponível)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"💾 Memória: {memory_mb:.1f} MB")
        except:
            pass
        
        print("="*70 + "\n")
    
    def _on_training_end(self) -> None:
        """Marca fim do treinamento."""
        if self.verbose >= 1:
            import time
            elapsed = time.time() - self.training_start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "="*70)
            print(" TREINAMENTO CONCLUÍDO")
            print("="*70)
            print(f"⏱️  Tempo total: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            print(f"📊 Total steps: {self.n_calls}")
            print("="*70 + "\n")


def create_callbacks(
    save_path: str = "./checkpoints",
    checkpoint_freq: int = 50000,
    log_freq: int = 1000,
    verbose: int = 1
) -> list:
    """Retorna callbacks padronizados com o conjunto reduzido de métricas."""

    os.makedirs(save_path, exist_ok=True)

    callbacks = [
        TrainingMonitorCallback(verbose=verbose, log_freq=log_freq * 10),
        BlockingRateCallback(verbose=verbose),
        RewardStatsCallback(verbose=verbose),
        FragmentationCallback(verbose=verbose),
        CustomCheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=save_path,
            name_prefix="ppo_qrmsa",
            save_best=True,
            verbose=verbose
        ),
    ]

    return callbacks


if __name__ == "__main__":
    """
    Demonstração dos callbacks (sem treinamento real).
    """
    print("\n" + "="*70)
    print(" CALLBACKS DISPONÍVEIS PARA MONITORAMENTO")
    print("="*70)
    
    callbacks_info = {
        "BlockingRateCallback": "Monitora taxa de bloqueio de serviços",
        "FragmentationCallback": "Analisa fragmentação espectral",
        "RewardStatsCallback": "Estatísticas detalhadas de rewards",
        "CustomCheckpointCallback": "Salva checkpoints e melhor modelo",
        "TrainingMonitorCallback": "Monitora progresso geral do treinamento"
    }
    
    for name, description in callbacks_info.items():
        print(f"\n📊 {name}")
        print(f"   {description}")
    
    print("\n" + "="*70)
    print("\n💡 Uso recomendado:")
    print("""
    callbacks = create_callbacks(
        save_path="./checkpoints",
        checkpoint_freq=50000,
        log_freq=1000,
        verbose=1
    )
    
    model.learn(
        total_timesteps=1000000,
        callback=callbacks
    )
    """)
    print("="*70 + "\n")
