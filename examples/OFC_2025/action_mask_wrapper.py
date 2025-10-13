#!/usr/bin/env python3
"""
ActionMaskWrapper - Wrapper otimizado para compatibilidade com MaskablePPO (SB3-Contrib)

Este wrapper expõe as máscaras de ações do ambiente QRMSA de forma eficiente
para uso com algoritmos que suportam máscaras de ações, como o MaskablePPO.

Características:
- Cache eficiente de máscaras para evitar recálculos
- Conversão otimizada para o formato esperado pelo SB3-Contrib
- Validação de consistência entre máscara e action space
- Overhead mínimo durante o treinamento

Autor: Criado para OFC 2025
Data: Outubro 2025
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper que expõe máscaras de ações para uso com MaskablePPO.
    
    O MaskablePPO espera um método `action_masks()` que retorna um array
    booleano indicando quais ações são válidas no estado atual.
    
    Este wrapper intercepta o `info` retornado pelo ambiente e armazena
    a máscara para acesso rápido pelo algoritmo de aprendizado.
    
    Exemplo de uso:
        >>> env = gym.make("QRMSAEnvWrapper-v0", **env_args)
        >>> env = ActionMaskWrapper(env)
        >>> model = MaskablePPO("MlpPolicy", env, ...)
        >>> model.learn(total_timesteps=1000000)
    """
    
    def __init__(self, env: gym.Env):
        """
        Inicializa o wrapper.
        
        Args:
            env: Ambiente Gymnasium a ser envolvido
        """
        super().__init__(env)
        
        # Cache da máscara atual
        self._current_mask: Optional[np.ndarray] = None
        
        # Flag para verificar se máscara foi atualizada
        self._mask_updated: bool = False
        
        # Validação do action space
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"ActionMaskWrapper only supports Discrete action spaces, "
                f"got {type(env.action_space)}"
            )
        
        self._action_space_n = env.action_space.n
        
        # Estatísticas (opcional - para debug/análise)
        self._total_steps = 0
        self._total_valid_actions = 0
        self._mask_cache_hits = 0
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset do ambiente e atualização da máscara inicial.
        
        Args:
            seed: Seed para reprodutibilidade
            options: Opções adicionais para o reset
            
        Returns:
            observation: Observação inicial
            info: Dicionário de informações (sem a máscara, que é acessada via action_masks())
        """
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Extrai e armazena a máscara do info
        self._update_mask_from_info(info)
        
        # Remove a máscara do info (opcional - pode manter se quiser)
        # O MaskablePPO usa action_masks(), então não precisa estar em info
        info_without_mask = {k: v for k, v in info.items() if k != 'mask'}
        
        return obs, info_without_mask
    
    def step(
        self, 
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executa um step no ambiente e atualiza a máscara.
        
        Args:
            action: Ação a ser executada
            
        Returns:
            observation: Nova observação
            reward: Recompensa recebida
            terminated: Se o episódio terminou
            truncated: Se o episódio foi truncado
            info: Dicionário de informações
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Atualiza estatísticas
        self._total_steps += 1
        
        # Extrai e armazena a máscara do info
        self._update_mask_from_info(info)
        
        # Remove a máscara do info (opcional)
        info_without_mask = {k: v for k, v in info.items() if k != 'mask'}
        
        return obs, reward, terminated, truncated, info_without_mask
    
    def action_masks(self) -> np.ndarray:
        """
        Retorna a máscara de ações válidas para o estado atual.
        
        Este método é chamado pelo MaskablePPO para obter as ações válidas.
        
        Returns:
            mask: Array booleano de shape (n_actions,) onde True indica ação válida
            
        Raises:
            RuntimeError: Se action_masks() for chamado antes de reset()
        """
        if self._current_mask is None:
            raise RuntimeError(
                "action_masks() called before reset(). "
                "Please call env.reset() first."
            )
        
        # Converte para booleano se necessário (pode estar como 0/1)
        if self._current_mask.dtype != bool:
            return self._current_mask.astype(bool)
        
        return self._current_mask
    
    def _update_mask_from_info(self, info: Dict[str, Any]) -> None:
        """
        Extrai a máscara do info e atualiza o cache.
        
        Args:
            info: Dicionário retornado pelo ambiente
            
        Raises:
            KeyError: Se 'mask' não estiver presente em info
            ValueError: Se a máscara tiver dimensões incorretas
        """
        if 'mask' not in info:
            raise KeyError(
                "Expected 'mask' key in info dictionary returned by environment. "
                "Make sure the environment returns action masks in the info dict."
            )
        
        mask = info['mask']
        
        # Validação de dimensões
        if len(mask) != self._action_space_n:
            raise ValueError(
                f"Mask length ({len(mask)}) does not match action space size "
                f"({self._action_space_n})"
            )
        
        # Validação de valores (deve ser 0/1 ou True/False)
        unique_values = np.unique(mask)
        if not all(v in [0, 1, True, False] for v in unique_values):
            raise ValueError(
                f"Mask contains invalid values. Expected 0/1 or True/False, "
                f"got {unique_values}"
            )
        
        # Atualiza cache
        self._current_mask = np.asarray(mask, dtype=bool)
        self._mask_updated = True
        
        # Atualiza estatísticas
        self._total_valid_actions += np.sum(self._current_mask)
    
    def get_mask_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre o uso de máscaras (útil para debug).
        
        Returns:
            stats: Dicionário com estatísticas de uso das máscaras
        """
        avg_valid_actions = (
            self._total_valid_actions / self._total_steps 
            if self._total_steps > 0 
            else 0
        )
        
        return {
            'total_steps': self._total_steps,
            'total_valid_actions': self._total_valid_actions,
            'avg_valid_actions_per_step': avg_valid_actions,
            'avg_valid_actions_ratio': avg_valid_actions / self._action_space_n if self._action_space_n > 0 else 0,
            'current_valid_actions': int(np.sum(self._current_mask)) if self._current_mask is not None else 0,
            'current_valid_ratio': float(np.mean(self._current_mask)) if self._current_mask is not None else 0.0,
        }
    
    def __repr__(self) -> str:
        """Representação em string do wrapper."""
        return f"<ActionMaskWrapper{self.env}>"


class ActionMaskMonitor(gym.Wrapper):
    """
    Wrapper adicional para monitorar estatísticas detalhadas de máscaras.
    
    Útil para análise e debug, mas pode ser removido em produção para
    melhor performance.
    """
    
    def __init__(self, env: gym.Env, log_frequency: int = 1000):
        """
        Inicializa o monitor.
        
        Args:
            env: Ambiente a ser monitorado
            log_frequency: A cada quantos steps imprimir estatísticas
        """
        super().__init__(env)
        self.log_frequency = log_frequency
        self._episode_masks = []
        self._episode_count = 0
        
    def reset(self, **kwargs):
        """Reset e logging de estatísticas do episódio anterior."""
        if self._episode_masks:
            self._log_episode_stats()
        
        self._episode_masks = []
        self._episode_count += 1
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step com coleta de estatísticas."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Coleta estatísticas da máscara
        if hasattr(self.env, 'action_masks'):
            mask = self.env.action_masks()
            self._episode_masks.append(mask)
        
        # Log periódico
        if len(self._episode_masks) % self.log_frequency == 0:
            self._log_current_stats()
        
        return obs, reward, terminated, truncated, info
    
    def _log_episode_stats(self):
        """Log estatísticas do episódio."""
        if not self._episode_masks:
            return
        
        masks_array = np.array(self._episode_masks)
        avg_valid = np.mean(np.sum(masks_array, axis=1))
        
        print(f"\n[Episode {self._episode_count}] Mask Statistics:")
        print(f"  Steps: {len(self._episode_masks)}")
        print(f"  Avg valid actions: {avg_valid:.2f} / {masks_array.shape[1]}")
        print(f"  Avg valid ratio: {avg_valid / masks_array.shape[1] * 100:.1f}%")
    
    def _log_current_stats(self):
        """Log estatísticas atuais."""
        if not self._episode_masks:
            return
        
        recent_masks = np.array(self._episode_masks[-self.log_frequency:])
        avg_valid = np.mean(np.sum(recent_masks, axis=1))
        
        print(f"[Step {len(self._episode_masks)}] Last {self.log_frequency} steps: "
              f"{avg_valid:.2f} valid actions on average")


# Função helper para facilitar o uso
def wrap_env_with_masks(env: gym.Env, monitor: bool = False) -> gym.Env:
    """
    Aplica o ActionMaskWrapper (e opcionalmente o monitor) ao ambiente.
    
    Args:
        env: Ambiente base
        monitor: Se True, adiciona também o ActionMaskMonitor
        
    Returns:
        env: Ambiente com wrappers aplicados
    """
    env = ActionMaskWrapper(env)
    
    if monitor:
        env = ActionMaskMonitor(env)
    
    return env
