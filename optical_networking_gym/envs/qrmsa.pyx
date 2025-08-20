from typing import Any, Literal, Sequence, SupportsFloat, Optional
from dataclasses import field

cimport cython
cimport numpy as cnp
from libc.stdint cimport uint32_t
from libc.math cimport log, exp, asinh, log10
cnp.import_array()

import gymnasium as gym
from gymnasium.utils import seeding
import functools
import heapq
import networkx as nx
import random
import numpy as np
from collections import defaultdict
from numpy.random import SeedSequence
from optical_networking_gym.utils import rle
from optical_networking_gym.core.osnr import calculate_osnr, calculate_osnr_observation
import math
import typing
import os
from scipy.signal import convolve

if typing.TYPE_CHECKING:
    from optical_networking_gym.topology import Link, Span, Modulation, Path

cdef class Band:
    """Classe para representar uma banda de frequência no espectro óptico"""
    cdef public str name
    cdef public double start_thz
    cdef public int num_slots
    cdef public double noise_figure_db
    cdef public double attenuation_db_km
    cdef public double slot_bw_hz
    
    # Atributos derivados
    cdef public double f_start_hz
    cdef public double f_end_hz
    cdef public int slot_start
    cdef public int slot_end
    
    def __init__(self, str name, double start_thz, int num_slots, 
                 double noise_figure_db, double attenuation_db_km, 
                 double slot_bw_hz = 12.5e9):
        # Validações
        if num_slots <= 0:
            raise ValueError(f"num_slots must be > 0, got {num_slots}")
        
        self.name = name
        self.start_thz = start_thz
        self.num_slots = num_slots
        self.noise_figure_db = noise_figure_db
        self.attenuation_db_km = attenuation_db_km
        self.slot_bw_hz = slot_bw_hz
        
        # Calcular atributos derivados
        self.f_start_hz = start_thz * 1e12
        self.f_end_hz = self.f_start_hz + self.slot_bw_hz * num_slots
        
        # Offsets serão definidos externamente via set_slot_offset
        self.slot_start = -1
        self.slot_end = -1
    
    cpdef void set_slot_offset(self, int offset):
        """Define o offset de slots na fibra (índices globais)"""
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        
        self.slot_start = offset
        self.slot_end = offset + self.num_slots
    
    cpdef bint contains_slot_range(self, int global_start, int length):
        """Verifica se um bloco de slots está totalmente dentro desta banda"""
        if self.slot_start < 0 or self.slot_end < 0:
            raise ValueError("Band slot offsets not set. Call set_slot_offset first.")
        
        cdef int global_end = global_start + length
        return global_start >= self.slot_start and global_end <= self.slot_end
    
    cpdef int global_to_local(self, int global_slot):
        """Converte índice global para local desta banda"""
        if self.slot_start < 0:
            raise ValueError("Band slot offsets not set. Call set_slot_offset first.")
        
        return global_slot - self.slot_start
    
    cpdef double center_frequency_hz_from_global(self, int global_start, int length):
        """Calcula frequência central de um bloco de slots (índices globais)"""
        if not self.contains_slot_range(global_start, length):
            raise ValueError(f"Slot range [{global_start}, {global_start + length}) not within band")
        
        cdef int local_start = self.global_to_local(global_start)
        cdef double center_slot = local_start + length / 2.0
        cdef double center_freq = self.f_start_hz + center_slot * self.slot_bw_hz
        
        return center_freq
    
    def __repr__(self):
        return (f"Band(name='{self.name}', start_thz={self.start_thz}, "
                f"num_slots={self.num_slots}, slot_range=[{self.slot_start}, {self.slot_end}), "
                f"noise_figure_db={self.noise_figure_db}, attenuation_db_km={self.attenuation_db_km})")

cdef class FastPathOps:
    """Operações vetorizadas rápidas para provisão e liberação de caminhos"""
    cdef TopologyCache topo_cache
    cdef int total_spectrum_slots  # Renomeado para ser mais claro
    
    def __init__(self, topo_cache: TopologyCache, total_spectrum_slots: int):
        self.topo_cache = topo_cache
        self.total_spectrum_slots = total_spectrum_slots
    
    cdef void fast_provision_path(self, cnp.int32_t[:] edge_indices, 
                                 int start_slot, int end_slot, 
                                 int service_id, object service):
        """Provisão vetorizada de caminho usando views de memória"""
        cdef int i, edge_idx
        cdef int num_edges = edge_indices.shape[0]
        
        # Operação vetorizada para atualizar available_slots
        for i in range(num_edges):
            edge_idx = edge_indices[i]
            self.topo_cache.available_slots[edge_idx, start_slot:end_slot] = 0
            
            # Adicionar serviço nas listas de edge
            self.topo_cache.edge_services[edge_idx].append(service)
            self.topo_cache.edge_running_services[edge_idx].append(service)
    
    cdef void fast_release_path(self, cnp.int32_t[:] edge_indices, 
                               int start_slot, int end_slot, object service):
        """Liberação vetorizada de caminho usando views de memória"""
        cdef int i, edge_idx
        cdef int num_edges = edge_indices.shape[0]
        
        # Operação vetorizada para liberar available_slots
        for i in range(num_edges):
            edge_idx = edge_indices[i]
            self.topo_cache.available_slots[edge_idx, start_slot:end_slot] = 1
            
            # Remover serviço das listas de edge
            try:
                self.topo_cache.edge_running_services[edge_idx].remove(service)
            except ValueError:
                pass  # Serviço já foi removido
    
    cdef cnp.ndarray[cnp.int32_t, ndim=1] extract_edge_indices(self, object path):
        """Extração rápida de índices de edges de um caminho"""
        cdef tuple node_list = path.node_list
        cdef int n = len(node_list) - 1
        cdef cnp.ndarray[cnp.int32_t, ndim=1] indices = np.empty(n, dtype=np.int32)
        cdef int i, node1_id, node2_id
        
        for i in range(n):
            node1_id = self.topo_cache.get_node_id(node_list[i])
            node2_id = self.topo_cache.get_node_id(node_list[i + 1])
            indices[i] = self.topo_cache.get_edge_index(node1_id, node2_id)
        
        return indices
    
    cdef bint batch_check_paths_free(self, list paths, int initial_slot, int number_slots):
        """Verificação em lote se múltiplos caminhos estão livres"""
        cdef int end = initial_slot + number_slots
        if end > self.total_spectrum_slots:
            return False
        
        cdef int start = initial_slot
        if end < self.total_spectrum_slots:
            end += 1
        
        cdef object path
        cdef cnp.ndarray[cnp.int32_t, ndim=1] edge_indices
        cdef int i, edge_idx
        
        for path in paths:
            edge_indices = self.extract_edge_indices(path)
            for i in range(edge_indices.shape[0]):
                edge_idx = edge_indices[i]
                if np.any(self.topo_cache.available_slots[edge_idx, start:end] == 0):
                    return False
        return True
    
    cdef cnp.ndarray[cnp.int32_t, ndim=1] vectorized_spectrum_product(self, cnp.ndarray[cnp.int32_t, ndim=1] edge_indices):
        """Produto vetorizado de espectros disponíveis para múltiplos links"""
        cdef int num_edges = edge_indices.shape[0]
        cdef int num_slots = self.total_spectrum_slots
        cdef cnp.ndarray[cnp.int32_t, ndim=2] available_matrix
        cdef cnp.ndarray[cnp.int32_t, ndim=1] result
        cdef int i, j
        
        available_matrix = self.topo_cache.available_slots[edge_indices, :]
        result = available_matrix[0].copy()
        
        # AND vetorizado otimizado
        for i in range(1, num_edges):
            for j in range(num_slots):
                result[j] *= available_matrix[i, j]
        
        return result

cdef class TopologyCache:
    """Cache para dados estáticos da topologia, evitando chamadas NetworkX durante simulação"""
    cdef public object edge_indices        # Índices dos links [num_edges]
    cdef public object edge_lengths        # Comprimentos dos links [num_edges]
    cdef public object edge_lookup         # [node_id][node_id] -> edge_index (-1 se não existe)
    cdef public list node_names            # Lista de nomes dos nós
    cdef public dict node_name_to_id       # Nome do nó -> ID
    cdef public dict node_id_to_name       # ID -> Nome do nó
    cdef public int num_nodes
    cdef public int num_edges
    cdef public object adjacency_matrix    # Matriz de adjacência [num_nodes][num_nodes]
    
    # Listas de serviços por link (dados dinâmicos)
    cdef public list edge_services         # Lista de listas: edge_services[edge_idx] = [services]
    cdef public list edge_running_services # Lista de listas: edge_running_services[edge_idx] = [running_services]
    
    # Cache para available_slots - referência direta ao array do NetworkX
    cdef public object available_slots     # Referência direta para topology.graph["available_slots"]
    cdef public object edge_data           # Lista de dicionários dos dados de cada edge
    
    def __init__(self, topology):
        """Inicializa cache extraindo dados do NetworkX"""
        self.num_nodes = topology.number_of_nodes()
        self.num_edges = topology.number_of_edges()
        
        # Cache de nós
        self.node_names = list(topology.nodes())
        self.node_name_to_id = {name: idx for idx, name in enumerate(self.node_names)}
        self.node_id_to_name = {idx: name for idx, name in enumerate(self.node_names)}
        
        # Inicializar arrays
        self.edge_indices = np.arange(self.num_edges, dtype=np.int32)
        self.edge_lengths = np.zeros(self.num_edges, dtype=np.float64)
        self.edge_lookup = np.full((self.num_nodes, self.num_nodes), -1, dtype=np.int32)
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int32)
        
        # Extrair informações dos links preservando ordem original
        for edge_idx, (node1, node2) in enumerate(topology.edges()):
            node1_id = self.node_name_to_id[node1]
            node2_id = self.node_name_to_id[node2]
            
            # Armazenar índice do link e comprimento
            edge_data = topology[node1][node2]
            
            # IMPORTANTE: Garantir que o índice no TopologyCache coincida com o do NetworkX
            topology_edge_index = edge_data.get('index', edge_idx)
            self.edge_lengths[topology_edge_index] = edge_data.get('length', 1.0)
            
            # Atualizar lookup bidirecionalmente (grafo não direcionado)
            self.edge_lookup[node1_id, node2_id] = topology_edge_index
            self.edge_lookup[node2_id, node1_id] = topology_edge_index
            
            # Atualizar matriz de adjacência
            self.adjacency_matrix[node1_id, node2_id] = 1
            self.adjacency_matrix[node2_id, node1_id] = 1
        
        # Inicializar listas de serviços por link
        self.edge_services = [[] for _ in range(self.num_edges)]
        self.edge_running_services = [[] for _ in range(self.num_edges)]
        
        # Referência direta para available_slots (será definida após criação do array no environment)
        self.available_slots = None
        
        # Cache de dados dos edges para acesso rápido 
        self.edge_data = []
        for edge_idx, (node1, node2) in enumerate(topology.edges()):
            edge_data = topology[node1][node2]
            self.edge_data.append(edge_data)
    
    cpdef int get_edge_index(self, int node1_id, int node2_id):
        """Retorna índice do link entre dois nós (-1 se não existe)"""
        return self.edge_lookup[node1_id, node2_id]
    
    cpdef double get_edge_length(self, int edge_idx):
        """Retorna comprimento do link"""
        return self.edge_lengths[edge_idx]
    
    cpdef int get_node_id(self, str node_name):
        """Retorna ID do nó pelo nome"""
        return self.node_name_to_id[node_name]
    
    cpdef str get_node_name(self, int node_id):
        """Retorna nome do nó pelo ID"""
        return self.node_id_to_name[node_id]
    
    cpdef void reset_services(self):
        """Limpa todas as listas de serviços"""
        for i in range(self.num_edges):
            self.edge_services[i].clear()
            self.edge_running_services[i].clear()
    
    cpdef list get_running_services(self, str node1_name, str node2_name):
        """Retorna lista de running_services para um link específico"""
        cdef int node1_id = self.get_node_id(node1_name)
        cdef int node2_id = self.get_node_id(node2_name)
        cdef int edge_idx = self.get_edge_index(node1_id, node2_id)
        if edge_idx >= 0:
            return self.edge_running_services[edge_idx]
        return []
    
    cpdef void set_available_slots_reference(self, object available_slots):
        """Define a referência direta para o array available_slots"""
        self.available_slots = available_slots

cdef class Service:
    cdef public int service_id
    cdef public str source
    cdef public int source_id
    cdef public object destination
    cdef public object destination_id
    cdef public float arrival_time
    cdef public float holding_time
    cdef public float bit_rate
    cdef public object path
    cdef public int service_class
    cdef public int initial_slot
    cdef public double center_frequency
    cdef public double bandwidth
    cdef public int number_slots
    cdef public int core
    cdef public double launch_power
    cdef public bint accepted
    cdef public bint blocked_due_to_resources
    cdef public bint blocked_due_to_osnr
    cdef public double OSNR
    cdef public double ASE
    cdef public double NLI
    cdef public object current_modulation
    cdef public object current_band  # Nova banda atual
    cdef public bint recalculate

    def __init__(
        self,
        int service_id,
        str source,
        int source_id,
        str destination = None,
        str destination_id = None,
        float arrival_time = 0.0,
        float holding_time = 0.0,
        float bit_rate = 0.0,
        object path = None,
        int service_class = 0,
        int initial_slot = 0,
        int center_frequency = 0,
        int bandwidth = 0,
        int number_slots = 0,
        int core = 0,
        double launch_power = 0.0,
        bint accepted = False,
        bint blocked_due_to_resources = True,
        bint blocked_due_to_osnr = True,
        float OSNR = 0.0,
        float ASE = 0.0,
        float NLI = 0.0,
        object current_modulation = None
    ):
        self.service_id = service_id
        self.source = source
        self.source_id = source_id
        self.destination = destination
        self.destination_id = destination_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.bit_rate = bit_rate
        self.path = path
        self.service_class = service_class
        self.initial_slot = initial_slot
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.number_slots = number_slots
        self.core = core
        self.launch_power = launch_power
        self.accepted = accepted
        self.blocked_due_to_resources = blocked_due_to_resources
        self.blocked_due_to_osnr = blocked_due_to_osnr
        self.OSNR = OSNR
        self.ASE = ASE
        self.NLI = NLI
        self.current_modulation = current_modulation
        self.current_band = None  # Inicializar como None
        self.recalculate = False

    def __repr__(self):
        return (
            f"Service(service_id={self.service_id}, source='{self.source}', source_id={self.source_id}, "
            f"destination='{self.destination}', destination_id={self.destination_id}, arrival_time={self.arrival_time}, "
            f"holding_time={self.holding_time}, bit_rate={self.bit_rate}, path={self.path}, service_class={self.service_class}, "
            f"initial_slot={self.initial_slot}, center_frequency={self.center_frequency}, bandwidth={self.bandwidth}, "
            f"number_slots={self.number_slots}, core={self.core}, launch_power={self.launch_power}, accepted={self.accepted}, "
            f"blocked_due_to_resources={self.blocked_due_to_resources}, blocked_due_to_osnr={self.blocked_due_to_osnr}, "
            f"OSNR={self.OSNR}, ASE={self.ASE}, NLI={self.NLI}, current_modulation={self.current_modulation}, "
            f"current_band={self.current_band}, recalculate={self.recalculate})"
        )

cdef class QRMSAEnv:
    cdef public uint32_t input_seed
    cdef public double load
    cdef int episode_length
    cdef double mean_service_holding_time
    cdef public int num_spectrum_resources
    cdef public double channel_width
    cdef bint allow_rejection
    cdef readonly object topology
    cdef TopologyCache topo_cache  # Novo cache de topologia
    cdef FastPathOps fast_ops      # Operações rápidas vetorizadas
    cdef readonly str bit_rate_selection
    cdef public tuple bit_rates
    cdef double bit_rate_lower_bound
    cdef double bit_rate_higher_bound
    cdef object bit_rate_probabilities
    cdef object node_request_probabilities
    cdef public object k_shortest_paths
    cdef public int k_paths
    cdef public double launch_power_dbm
    cdef public double launch_power
    cdef double bandwidth
    cdef public double frequency_start
    cdef public double frequency_end
    cdef public double frequency_slot_bandwidth
    cdef public double margin
    cdef public object modulations
    cdef bint measure_disruptions
    cdef public object _np_random
    cdef public int _np_random_seed
    cdef object spectrum_use
    cdef object spectrum_allocation
    cdef public Service current_service
    cdef int service_id_counter
    cdef list services_in_progress
    cdef list release_times
    cdef int services_processed
    cdef int services_accepted
    cdef int episode_services_processed
    cdef int episode_services_accepted
    cdef double bit_rate_requested
    cdef double bit_rate_provisioned
    cdef double episode_bit_rate_requested
    cdef double episode_bit_rate_provisioned
    cdef object bit_rate_requested_histogram
    cdef object bit_rate_provisioned_histogram
    cdef object slots_provisioned_histogram
    cdef object episode_slots_provisioned_histogram
    cdef int disrupted_services
    cdef int episode_disrupted_services
    cdef list disrupted_services_list
    cdef public object action_space
    cdef public object observation_space
    cdef object episode_actions_output
    cdef object episode_actions_taken
    cdef object episode_modulation_histogram
    cdef object episode_bit_rate_requested_histogram
    cdef object episode_bit_rate_provisioned_histogram
    cdef object spectrum_slots_allocation
    cdef public int reject_action
    cdef object actions_output
    cdef object actions_taken
    cdef bint _new_service
    cdef public double current_time
    cdef double mean_service_inter_arrival_time
    cdef public object frequency_vector
    cdef object rng
    cdef object bit_rate_function
    cdef list _events
    cdef object file_stats
    cdef unicode final_file_name
    cdef int blocks_to_consider
    cdef int bl_resource 
    cdef int bl_osnr 
    cdef int bl_reject
    cdef public int max_modulation_idx
    cdef public int modulations_to_consider
    cdef int spectrum_efficiency_metric
    cdef bint defragmentation
    cdef int n_defrag_services
    cdef int episode_defrag_cicles
    cdef int episode_service_realocations
    cdef bint gen_observation
    
    # Atributos para multibanda
    cdef public list bands
    cdef public int num_bands
    cdef public int total_slots

    topology: cython.declare(nx.Graph, visibility="readonly")
    bit_rate_selection: cython.declare(Literal["continuous", "discrete"], visibility="readonly")
    bit_rates: cython.declare(tuple[int, int, int] or tuple[float, float, float], visibility="readonly")

    def __init__(
        self,
        topology: nx.Graph,
        band_specs,  # Obrigatório - lista de especificações de bandas
        episode_length: int = 1000,
        load: float = 10.0,
        mean_service_holding_time: double = 10800.0,
        bit_rate_selection: str = "continuous",
        bit_rates: tuple = (10, 40, 100),
        bit_rate_probabilities = None,
        node_request_probabilities = None,
        bit_rate_lower_bound: float = 25.0,
        bit_rate_higher_bound: float = 100.0,
        launch_power_dbm: float = 0.0,
        bandwidth: float = 4e12,
        frequency_start: float = (3e8 / 1565e-9),
        frequency_slot_bandwidth: float = 12.5e9,
        margin: float = 0.0,
        measure_disruptions: bool = False,
        seed: object = None,
        allow_rejection: bool = True,
        reset: bool = True,
        channel_width: double = 12.5,
        k_paths: int = 5,
        file_name: str = "",
        blocks_to_consider: int = 1,
        modulations_to_consider: int = 6,
        defragmentation: bool = False,
        n_defrag_services: int = 0,
        gen_observation: bool = True,
    ):
        self.gen_observation = gen_observation
        self.defragmentation = defragmentation
        self.n_defrag_services = n_defrag_services
        self.rng = random.Random()
        self.blocks_to_consider = blocks_to_consider
        self.mean_service_inter_arrival_time = 0
        self.set_load(load=load, mean_service_holding_time=mean_service_holding_time)
        self.bit_rate_selection = bit_rate_selection

        if self.bit_rate_selection == "continuous":
            self.bit_rate_lower_bound = bit_rate_lower_bound
            self.bit_rate_higher_bound = bit_rate_higher_bound
            self.bit_rate_function = functools.partial(
                self.rng.randint,
                int(self.bit_rate_lower_bound),
                int(self.bit_rate_higher_bound)
            )
        elif self.bit_rate_selection == "discrete":
            if bit_rate_probabilities is None:
                bit_rate_probabilities = [1.0 / len(bit_rates) for _ in range(len(bit_rates))]
            self.bit_rate_probabilities = bit_rate_probabilities
            self.bit_rates = bit_rates
            self.bit_rate_function = functools.partial(
                self.rng.choices, self.bit_rates, self.bit_rate_probabilities, k=1
            )
            self.bit_rate_requested_histogram = defaultdict(int)
            self.bit_rate_provisioned_histogram = defaultdict(int)
            self.episode_bit_rate_requested_histogram = defaultdict(int)
            self.episode_bit_rate_provisioned_histogram = defaultdict(int)
            self.slots_provisioned_histogram = defaultdict(int)
            self.episode_slots_provisioned_histogram = defaultdict(int)
        self.topology = topology
        # Criar cache de topologia para acelerar acesso aos dados
        self.topo_cache = TopologyCache(topology)
        
        # Configurar basic parameters necessários para _init_bands
        self.k_paths = k_paths
        self.modulations = self.topology.graph.get("modulations", [])
        self.max_modulation_idx = len(self.modulations) - 1
        self.modulations_to_consider = min(modulations_to_consider, len(self.modulations))
        self.frequency_slot_bandwidth = frequency_slot_bandwidth  # Necessário antes de _init_bands
        
        # Verificar se band_specs foi fornecido
        if band_specs is None:
            raise ValueError("band_specs is required. Must provide a list of band specifications.")
        
        # Inicializar bandas - num_spectrum_resources será calculado automaticamente
        self._init_bands(band_specs, 0, frequency_start)  # 0 é ignorado agora
        
        # Inicializar operações rápidas vetorizadas (usar total_slots agora)
        self.fast_ops = FastPathOps(self.topo_cache, self.total_slots)
        self.episode_length = episode_length
        self.load = load
        self.mean_service_holding_time = mean_service_holding_time
        self.channel_width = channel_width
        self.allow_rejection = allow_rejection
        self.k_shortest_paths = self.topology.graph["ksp"]
        if node_request_probabilities is not None:
            self.node_request_probabilities = node_request_probabilities
        else:
            # Usar cache em vez de topology.number_of_nodes()
            tmp_probabilities = np.full(
                (self.topo_cache.num_nodes,),
                fill_value=1.0 / self.topo_cache.num_nodes,
                dtype=np.float64
            )
            self.node_request_probabilities = np.asarray(tmp_probabilities, dtype=np.float64)
        self.launch_power_dbm = launch_power_dbm
        self.launch_power = 10 ** ((self.launch_power_dbm - 30) / 10)
        self.bandwidth = bandwidth
        self.frequency_start = frequency_start
        # frequency_slot_bandwidth já foi definido antes de _init_bands
        self.margin = margin
        self.measure_disruptions = measure_disruptions
        self.frequency_end = self.frequency_start + (self.frequency_slot_bandwidth * self.num_spectrum_resources)
        assert math.isclose(self.frequency_end - self.frequency_start, self.bandwidth, rel_tol=1e-5)
        self.frequency_vector = np.linspace(
            self.frequency_start,
            self.frequency_end,
            num=self.num_spectrum_resources,
            dtype=np.float64
        )
        assert self.frequency_vector.shape[0] == self.num_spectrum_resources, (
            f"Size of frequency_vector ({self.frequency_vector.shape[0]}) "
            f"does not match num_spectrum_resources ({self.num_spectrum_resources})."
        )
        self.topology.graph["available_slots"] = np.ones(
            (self.topo_cache.num_edges, self.num_spectrum_resources),
            dtype=np.int32
        )
        # Definir referência direta no cache
        self.topo_cache.set_available_slots_reference(self.topology.graph["available_slots"])
       
        self.disrupted_services_list = []
        self.disrupted_services = 0
        self.episode_disrupted_services = 0

        # O action_space é criado em _init_bands() após definir bandas

        total_dim = (
            1
            + 2
            + self.k_paths
            + (self.k_paths * self.num_bands * self.modulations_to_consider * 12)  # Usar num_bands
        )

        self.observation_space = gym.spaces.Box(
                low=-5,
                high=5,
                shape=(total_dim,),
                dtype=np.float32
            )
        if seed is None:
            ss = SeedSequence()
            input_seed = int(ss.generate_state(1)[0])
        elif isinstance(seed, int):
            input_seed = int(seed)
        else:
            raise ValueError("Seed must be an integer.")
        input_seed = input_seed % (2 ** 31)
        if input_seed >= 2 ** 31:
            input_seed -= 2 ** 32
        self.input_seed = int(input_seed)
        self._np_random, self._np_random_seed = seeding.np_random(self.input_seed)
        num_edges = self.topo_cache.num_edges  # Usar cache em vez de NetworkX
        
        # Usar total_slots para arrays de espectro
        self.spectrum_use = np.zeros(
            (num_edges, self.total_slots), dtype=np.int32
        )
        self.spectrum_allocation = np.full(
            (num_edges, self.total_slots),
            fill_value=-1,
            dtype=np.int64
        )
        self.current_service = None
        self.service_id_counter = 0
        self.services_in_progress = []
        self.release_times = []
        self.current_time = 0.0
        self._events = []
        self.services_processed = 0
        self.services_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.bit_rate_requested = 0.0
        self.bit_rate_provisioned = 0.0
        self.episode_bit_rate_requested = 0.0
        self.episode_bit_rate_provisioned = 0.0
        if self.bit_rate_selection == "discrete":
            self.bit_rate_requested_histogram = defaultdict(int)
            self.bit_rate_provisioned_histogram = defaultdict(int)
            self.slots_provisioned_histogram = defaultdict(int)
            self.episode_slots_provisioned_histogram = defaultdict(int)
        else:
            self.bit_rate_requested_histogram = None
            self.bit_rate_provisioned_histogram = None
        self.reject_action = self.action_space.n - 1 if allow_rejection else 0
        self.actions_output = np.zeros(
            (self.k_paths + 1, self.total_slots + 1), dtype=np.int64  # Usar total_slots
        )
        self.actions_taken = np.zeros(
            (self.k_paths + 1, self.total_slots + 1), dtype=np.int64  # Usar total_slots
        )
        if file_name != "":
            final_name = "_".join([
                file_name,
                str(self.topology.graph["name"]),
                str(self.launch_power_dbm),
                str(self.load),
                str(seed) + ".csv"
            ])

            dir_name = os.path.dirname(final_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            self.final_file_name = final_name
            self.file_stats = open(final_name, "wt", encoding="UTF-8")

            self.file_stats.write("# Service stats file from simulator\n")
            self.file_stats.write("id,source,destination,bit_rate,path_k,path_length,modulation,min_osnr,osnr,ase,nli,disrupted_services,active_services\n")
        else:
            self.file_stats = None

        self.bl_osnr = 0
        self.bl_resource = 0
        self.bl_reject = 0
        self.spectrum_efficiency_metric = 0
        self.episode_defrag_cicles = 0
        self.episode_service_realocations = 0
        if reset:
            self.reset()

    def _init_bands(self, object band_specs, int unused_param, double frequency_start):
        """Inicializa bandas de frequência a partir de band_specs (obrigatório)."""
        print(f"[DEBUG] _init_bands: Iniciando com {len(band_specs)} band_specs")
        cdef list bands = []
        cdef Band band
        cdef dict spec
        cdef int current_offset = 0
        cdef int i, j
        cdef double prev_f_end = 0.0
        
        # Calcular num_spectrum_resources automaticamente baseado nas bandas
        calculated_num_spectrum_resources = sum(
            spec["num_slots"] if isinstance(spec, dict) else spec.num_slots 
            for spec in band_specs
        )
        print(f"[DEBUG] _init_bands: Calculado num_spectrum_resources = {calculated_num_spectrum_resources}")
        self.num_spectrum_resources = calculated_num_spectrum_resources
        
        # Converter specs para objetos Band se necessário
        for spec in band_specs:
            if isinstance(spec, Band):
                bands.append(spec)
            elif isinstance(spec, dict):
                required_keys = {"name", "start_thz", "num_slots", "noise_figure_db", "attenuation_db_km"}
                if not required_keys.issubset(spec.keys()):
                    missing = required_keys - spec.keys()
                    raise ValueError(f"Missing keys in band_specs: {missing}")
                
                band = Band(
                    name=spec["name"],
                    start_thz=spec["start_thz"],
                    num_slots=spec["num_slots"],
                    noise_figure_db=spec["noise_figure_db"],
                    attenuation_db_km=spec["attenuation_db_km"],
                    slot_bw_hz=self.frequency_slot_bandwidth
                )
                print(f"[DEBUG] _init_bands: Criada banda {band.name}: {spec['start_thz']} THz, {spec['num_slots']} slots")
                bands.append(band)
            else:
                    raise ValueError(f"band_specs must contain Band objects or dicts, got {type(spec)}")
        
        # Ordenar bandas por frequência inicial
        bands.sort(key=lambda b: b.f_start_hz)
        print(f"[DEBUG] _init_bands: Bandas ordenadas por frequência")
        
        # Verificar sobreposições e atribuir offsets contíguos
        for i, band in enumerate(bands):
            if i > 0:
                prev_band = bands[i-1]
                if prev_band.f_end_hz > band.f_start_hz:
                    raise ValueError(
                        f"Band overlap detected: {prev_band.name} ends at {prev_band.f_end_hz/1e12:.2f} THz, "
                        f"but {band.name} starts at {band.f_start_hz/1e12:.2f} THz"
                    )
            
            band.set_slot_offset(current_offset)
            print(f"[DEBUG] _init_bands: Banda {band.name} offset {current_offset}, slots [{band.slot_start}-{band.slot_end})")
            current_offset += band.num_slots
        
        # Atribuir às variáveis de instância
        self.bands = bands
        self.num_bands = len(bands)
        self.total_slots = current_offset
        
        print(f"[DEBUG] _init_bands: Finalizado - {self.num_bands} bandas, {self.total_slots} slots totais")
        
        # Atualizar action_space para usar slots por banda (não total)
        # Assumindo bandas uniformes de 10 slots cada
        slots_per_band = self.bands[0].num_slots  # Usar primeira banda como referência
        action_space_size = (self.k_paths * self.num_bands * self.modulations_to_consider * slots_per_band) + 1
        print(f"[DEBUG] _init_bands: Action space = {self.k_paths} × {self.num_bands} × {self.modulations_to_consider} × {slots_per_band} + 1 = {action_space_size}")
        self.action_space = gym.spaces.Discrete(action_space_size)

    cpdef Band band_for_global_slot(self, int global_slot):
        """Retorna a banda que contém o slot global especificado"""
        print(f"[DEBUG] band_for_global_slot: Procurando banda para slot global {global_slot}")
        cdef Band band
        for band in self.bands:
            print(f"[DEBUG] band_for_global_slot: Verificando banda {band.name} [{band.slot_start}, {band.slot_end})")
            if band.slot_start <= global_slot < band.slot_end:
                print(f"[DEBUG] band_for_global_slot: ✓ Slot {global_slot} encontrado na banda {band.name}")
                return band
        print(f"[DEBUG] band_for_global_slot: ✗ Nenhuma banda contém o slot {global_slot}")
        raise ValueError(f"No band contains global slot {global_slot}")

    cpdef tuple reset(self, object seed=None, dict options=None):
        self.episode_bit_rate_requested = 0.0
        self.episode_bit_rate_provisioned = 0.0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_disrupted_services = 0
        self._events = []
        self.bl_resource = 0
        self.bl_osnr = 0
        self.bl_reject = 0
        self.max_modulation_idx = len(self.modulations) - 1
        self.episode_defrag_cicles = 0
        self.episode_service_realocations = 0

        self.episode_actions_output = np.zeros(
            (self.k_paths + self.reject_action, self.total_slots + self.reject_action),  # Usar total_slots
            dtype=np.int32
        )
        self.episode_actions_taken = np.zeros(
            (self.k_paths + self.reject_action, self.total_slots + self.reject_action),  # Usar total_slots
            dtype=np.int32
        )

        if self.bit_rate_selection == "discrete":
            self.episode_bit_rate_requested_histogram = {}
            self.episode_bit_rate_provisioned_histogram = {}
            for bit_rate in self.bit_rates:
                self.episode_bit_rate_requested_histogram[bit_rate] = 0
                self.episode_bit_rate_provisioned_histogram[bit_rate] = 0

        self.episode_modulation_histogram = {}
        for modulation in self.modulations:
            self.episode_modulation_histogram[modulation.spectral_efficiency] = 0

        if options is not None and "only_episode_counters" in options and options["only_episode_counters"]:
            observation, mask = self.observation()
            info = {}
            return observation, info

        self.bit_rate_requested = 0.0
        self.bit_rate_provisioned = 0.0
        self.disrupted_services = 0
        self.disrupted_services_list = []

        self.topology.graph["services"] = []
        self.topology.graph["running_services"] = []
        self.topology.graph["last_update"] = 0.0

        # Usar cache para limpar listas de serviços
        for edge_idx in range(self.topo_cache.num_edges):
            self.topo_cache.edge_services[edge_idx] = []
            self.topo_cache.edge_running_services[edge_idx] = []

        self.topology.graph["available_slots"] = np.ones(
            (self.topo_cache.num_edges, self.total_slots),  # Usar total_slots
            dtype=np.int32
        )
        # Atualizar referência direta no cache
        self.topo_cache.set_available_slots_reference(self.topology.graph["available_slots"])

        self.spectrum_slots_allocation = np.full(
            (self.topo_cache.num_edges, self.total_slots),  # Usar total_slots
            fill_value=-1,
            dtype=np.int32
        )

        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0

        # Inicializar estatísticas dos links usando cache
        for edge_idx in range(self.topo_cache.num_edges):
            edge_data = self.topo_cache.edge_data[edge_idx]
            edge_data["external_fragmentation"] = 0.0
            edge_data["compactness"] = 0.0
            edge_data["utilization"] = 0.0
            edge_data["last_update"] = 0.0

        self._new_service = False
        self._next_service()

        observation, mask = self.observation()
        info = mask.copy()
        return observation, info
    
    cpdef public normalize_value(self, value, min_v, max_v):
        """
        Normaliza um valor no intervalo [0,1]. 
        Se max_v == min_v, retorna 0 para evitar divisão por zero.
        """
        if max_v == min_v:
            return 0.0
        return (value - min_v) / (max_v - min_v)
    
    cpdef public _get_candidates(self, available_slots, num_slots_required, total_slots):
        """
        Gera todos os candidatos (initial slot) para alocação, usando RLE.
        Se o bloco se estende até o final, não exige guard band; caso contrário,
        exige num_slots_required+1 slots.
        
        Args:
            available_slots (np.array): Vetor com slots disponíveis (1) e indisponíveis (0).
            num_slots_required (int): Número de slots necessários para o serviço.
            total_slots (int): Número total de slots no espectro.
        
        Returns:
            list: Lista de candidatos (índices) válidos.
        """
        initial_indices, values, lengths = rle(available_slots)
        candidates = []
        for start, val, length in zip(initial_indices, values, lengths):
            if val == 1:
                if start + length == total_slots:
                    if length >= num_slots_required:
                        for candidate in range(start, start + length - num_slots_required + 1):
                            candidates.append(candidate)
                else:
                    if length >= (num_slots_required + 1):
                        for candidate in range(start, start + length - (num_slots_required + 1) + 1):
                            candidates.append(candidate)
        return candidates

    cpdef _get_candidates_in_band(self, available_slots, int num_slots_required, Band band):
        """
        Gera candidatos para alocação dentro de uma banda específica.
        Não permite cruzar fronteiras de banda.
        
        Args:
            available_slots (np.array): Vetor global com slots disponíveis.
            num_slots_required (int): Número de slots necessários.
            band (Band): Banda onde buscar candidatos.
        
        Returns:
            list: Lista de candidatos (índices globais) válidos.
        """
        
        # Extrair apenas a janela da banda
        band_available = available_slots[band.slot_start:band.slot_end]
        
        # Usar a lógica existente na janela da banda
        initial_indices, values, lengths = rle(band_available)
        candidates = []
        
        for start, val, length in zip(initial_indices, values, lengths):
            if val == 1:
                # Verificar se o bloco vai até o final da BANDA (não do espectro total)
                is_end_of_band = (start + length == band.num_slots)
                
                if is_end_of_band:
                    # Final da banda: não exige guard band
                    if length >= num_slots_required:
                        range_start = start
                        range_end = start + length - num_slots_required + 1
                        
                        for candidate in range(range_start, range_end):
                            # Converter para índice global
                            global_candidate = band.slot_start + candidate
                            candidates.append(global_candidate)
                else:
                    # Meio da banda: exige guard band
                    if length >= (num_slots_required + 1):
                        range_start = start
                        range_end = start + length - (num_slots_required + 1) + 1
                        
                        for candidate in range(range_start, range_end):
                            # Converter para índice global
                            global_candidate = band.slot_start + candidate
                            candidates.append(global_candidate)
        
        return candidates

    cpdef public get_max_modulation_index(self):
        """
        Atualiza self.max_modulation_idx considerando (path × banda × modulation).
        Usa candidatos por banda e OSNR de observação com NF/α da banda.
        """
        for path in self.k_shortest_paths[self.current_service.source, self.current_service.destination]:
            available_slots = self.get_available_slots(path)
            
            for band in self.bands:
                for idm, modulation in enumerate(reversed(self.modulations)):
                    number_slots = self.get_number_slots(self.current_service, modulation)
                    
                    # Usar candidatos dentro da banda
                    candidatos = self._get_candidates_in_band(available_slots, number_slots, band)
                    
                    if candidatos:
                        for candidate in candidatos:
                            # Calcular centro de frequência via banda
                            center_frequency = band.center_frequency_hz_from_global(candidate, number_slots)
                            
                            # Usar OSNR de observação com parâmetros da banda
                            osnr_obs = calculate_osnr_observation(
                                self,
                                path.links,
                                self.frequency_slot_bandwidth * number_slots,  # bandwidth
                                center_frequency,
                                self.current_service.service_id,
                                self.launch_power,
                                modulation.minimum_osnr + self.margin,
                                band  # Passar banda para usar NF/α específicos
                            )

                            if osnr_obs >= 0:  # OSNR aceitável
                                self.max_modulation_idx = max(len(self.modulations) - idm - 1,
                                                            self.modulations_to_consider - 1)
                                return
        
        # Se nenhuma modulação funcionou, usar a mais conservadora
        self.max_modulation_idx = self.modulations_to_consider - 1

    def observation(self):
        if not self.gen_observation:
            obs    = np.zeros((self.observation_space.shape[0],), dtype=np.float32)
            action_mask = np.zeros((self.action_space.n,),           dtype=np.uint8)
            return obs, {'mask': action_mask}
            
        def compute_modulation_features(available_slots, num_slots_required, route_links, modulation, band):
            # Usar candidatos específicos da banda ao invés de candidatos globais
            valid_starts = self._get_candidates_in_band(available_slots, num_slots_required, band)
            candidate_count = len(valid_starts)
            feature_candidate_count = candidate_count / band.num_slots  # Normalizar pela banda, não pelo total
            
            if candidate_count > 0:
                # Converter para slots locais da banda para cálculos de features
                local_starts = [slot - band.slot_start for slot in valid_starts]
                avg_candidate = np.mean(local_starts)
                std_candidate = np.std(local_starts)
                max_candidate = max(local_starts)
            else:
                avg_candidate = std_candidate = max_candidate = 0.0
            
            feature_avg_candidate = avg_candidate / (band.num_slots - 1) if band.num_slots > 1 else 0.0
            feature_std_candidate = std_candidate / (band.num_slots - 1) if band.num_slots > 1 else 0.0
            feature_max_candidate = max_candidate / (band.num_slots - 1) if band.num_slots > 1 else 0.0
            
            osnr_values = []
            osnr_best = 0.0
            for init_slot in valid_starts:
                service_bandwidth = num_slots_required * self.channel_width * 1e9
                # Usar banda para calcular frequência central corretamente
                service_center_frequency = band.center_frequency_hz_from_global(init_slot, num_slots_required)
                
                osnr_current = calculate_osnr_observation(
                    self,
                    route_links,
                    service_bandwidth,
                    service_center_frequency,
                    self.current_service.service_id,
                    10 ** ((self.launch_power_dbm - 30) / 10),
                    modulation.minimum_osnr
                )
                osnr_values.append(osnr_current)
                if osnr_current > osnr_best:
                    osnr_best = osnr_current
            if osnr_values:
                osnr_mean = np.mean(osnr_values)
                osnr_var = np.var(osnr_values)
            else:
                osnr_mean = osnr_var = 0.0
            
            adjusted_slots_required = max((num_slots_required - 5.5) / 3.5, 0.0)
            
            # Usar slots disponíveis da banda, não globais
            band_available = available_slots[band.slot_start:band.slot_end]
            total_available_slots = np.sum(band_available)
            total_available_slots_ratio = 2.0 * (total_available_slots - 0.5 * band.num_slots) / band.num_slots
            
            blocks_sizes = []
            current_len = 0
            for slot in band_available:
                if slot == 1:
                    current_len += 1
                else:
                    if current_len > 0:
                        blocks_sizes.append(current_len)
                    current_len = 0
            if current_len > 0:
                blocks_sizes.append(current_len)
            if blocks_sizes:
                mean_block_size = ((np.mean(blocks_sizes) - 4.0) / 4.0) / 100.0
                std_block_size = (np.std(blocks_sizes) / 100.0)
            else:
                mean_block_size = std_block_size = 0.0
            
            link_usage_normalized = 2.0 * ((np.sum(band_available) / band.num_slots) - 0.5)
            
            features = [feature_candidate_count,
                        feature_avg_candidate,
                        feature_std_candidate,
                        adjusted_slots_required,
                        total_available_slots_ratio,
                        mean_block_size,
                        std_block_size,
                        osnr_best,
                        osnr_mean,
                        osnr_var,
                        link_usage_normalized,
                        feature_max_candidate]
            return valid_starts, features, osnr_values

        # ========================
        # Observações comuns
        # ========================
        topology = self.topology
        current_service = self.current_service
        num_spectrum_resources = self.num_spectrum_resources
        k_shortest_paths = self.k_shortest_paths
        modulations = self.modulations
        num_mod_to_consider = self.modulations_to_consider
        num_nodes = self.topo_cache.num_nodes  # Usar cache
        frequency_slot_bandwidth = self.channel_width * 1e9
        max_bit_rate = max(self.bit_rates)
        self.get_max_modulation_index()

        source_id = int(current_service.source_id)
        destination_id = int(current_service.destination_id)
        source_norm = source_id / (num_nodes - 1) if num_nodes > 1 else 0
        destination_norm = destination_id / (num_nodes - 1) if num_nodes > 1 else 0
        source_destination = np.array([source_norm, destination_norm], dtype=np.float32)

        bit_rate_obs = np.array([current_service.bit_rate / max_bit_rate], dtype=np.float32)

        num_paths_to_evaluate = self.k_paths
        # Usar comprimentos dos links do cache para normalização das rotas
        min_length = np.min(self.topo_cache.edge_lengths)
        max_length = np.max(self.topo_cache.edge_lengths)
        route_lengths = np.zeros((num_paths_to_evaluate,), dtype=np.float32)

        # Pré-cálculo das informações de cada caminho: (route, available_slots)
        paths_info = []
        source = current_service.source
        destination = current_service.destination
        for path_index, route in enumerate(k_shortest_paths[source, destination]):
            if path_index >= num_paths_to_evaluate:
                break
            normalized_length = self.normalize_value(route.length, min_length, max_length)
            route_lengths[path_index] = normalized_length
            available_slots = self.get_available_slots(route)
            paths_info.append((route, available_slots))

        # ========================
        # Cálculo das features de observação por (caminho, banda, modulação)
        # ========================
        mod_features_obs = np.full((num_paths_to_evaluate * self.num_bands * num_mod_to_consider, 12), fill_value=-1.0, dtype=np.float32)
        mod_features_cache = {}
        # Include service bit rate in cache key to avoid sharing between different services
        service_bit_rate = current_service.bit_rate
        
        for p_idx in range(num_paths_to_evaluate):
            route, available_slots = paths_info[p_idx]
            start_index = 0 if self.max_modulation_idx <= 1 else max(0, self.max_modulation_idx - (num_mod_to_consider - 1))
            mod_list = list(reversed(modulations[start_index: num_mod_to_consider + start_index]))
            
            for band_idx, band in enumerate(self.bands):
                for m_idx in range(num_mod_to_consider):
                    modulation = mod_list[m_idx]
                    num_slots_required = self.get_number_slots(current_service, modulation)
                    
                    # Calcular features específicas para esta banda
                    valid_starts, features, osnr_values = compute_modulation_features(
                        available_slots, num_slots_required, route.links, modulation, band
                    )
                    
                    # Índice único para (path, band, modulation)
                    feature_idx = p_idx * self.num_bands * num_mod_to_consider + band_idx * num_mod_to_consider + m_idx
                    mod_features_obs[feature_idx, :] = np.array(features, dtype=np.float32)
                    
                    # Cache com chave (path, band, modulation, bit_rate)
                    mod_features_cache[(p_idx, band_idx, m_idx, service_bit_rate)] = (valid_starts, features, num_slots_required, modulation)

        # ========================
        # Geração da máscara de ações (atualizada para multibanda)
        # ========================
        total_actions = self.action_space.n - 1  # Excluindo ação dummy
        action_mask = np.zeros(total_actions + 1, dtype=np.uint8)

        for action_index in range(total_actions):
            # Decodificar ação usando slots por banda (não total)
            slots_per_band = self.bands[0].num_slots
            decoded = self.decimal_to_array(
                action_index,
                [self.k_paths, self.num_bands, self.modulations_to_consider, slots_per_band]
            )
            p_idx, band_idx, modulation_idx, slot_relative = decoded
            
            # Converter slot relativo para slot global dentro da banda
            band = self.bands[band_idx]
            init_slot = band.slot_start + slot_relative

            # Verificar se índices são válidos
            if (p_idx >= num_paths_to_evaluate or 
                band_idx >= len(self.bands) or 
                modulation_idx >= len(self.modulations)):
                continue

            # Converter modulation_idx de volta para índice relativo para acessar cache
            if self.max_modulation_idx > 1:
                allowed_mods = list(range(self.max_modulation_idx, self.max_modulation_idx - self.modulations_to_consider, -1))
            else:
                allowed_mods = list(range(0, self.modulations_to_consider))
            
            try:
                m_idx_relative = allowed_mods.index(modulation_idx)
            except ValueError:
                # Modulação inválida
                continue

            # Recupera os dados de modulação do cache usando chave (path, band, modulation, bit_rate)
            route, available_slots = paths_info[p_idx]
            band = self.bands[band_idx]
            valid_starts, features, num_slots_required, modulation = mod_features_cache[(p_idx, band_idx, m_idx_relative, service_bit_rate)]

            valid_action = False
            osnr_current = 0.0
            
            # Verificar se slot está dentro da banda e se existe no conjunto de starts válidos
            slot_in_band = (init_slot >= band.slot_start and init_slot + num_slots_required <= band.slot_end)
            slot_in_valid_starts = (init_slot in valid_starts)
            
            if slot_in_band and slot_in_valid_starts:
                
                service_bandwidth = num_slots_required * frequency_slot_bandwidth
                # Usar banda para calcular frequência central
                service_center_frequency = band.center_frequency_hz_from_global(init_slot, num_slots_required)
                
                osnr_current = calculate_osnr_observation(
                    self,
                    route.links,
                    service_bandwidth,
                    service_center_frequency,
                    current_service.service_id,
                    10 ** ((self.launch_power_dbm - 30) / 10),
                    modulation.minimum_osnr
                )
                
                if osnr_current >= 0:
                    valid_action = True

            # DEBUG: Print resumido para todas as ações (removendo filtro)
            #if True:  # Mostrar todas as ações
                #if valid_action:
                    #print(f"[MASK DEBUG] Ação {action_index}: ✅ VÁLIDA - slot {slot_relative}, {modulation.name}, band {band.name}, path {p_idx}, OSNR={osnr_current:.2f}")
                #else:
                    #if not slot_in_band:
                        #print(f"[MASK DEBUG] Ação {action_index}: ❌ INVÁLIDA - slot {slot_relative}, {modulation.name}, band {band.name}, path {p_idx} - slot fora da banda")
                    #elif not slot_in_valid_starts:
                        #print(f"[MASK DEBUG] Ação {action_index}: ❌ INVÁLIDA - slot {slot_relative}, {modulation.name}, band {band.name}, path {p_idx} - slot não está em valid_starts")
                    #else:
                        #print(f"[MASK DEBUG] Ação {action_index}: ❌ INVÁLIDA - slot {slot_relative}, {modulation.name}, band {band.name}, path {p_idx} - OSNR insuficiente ({osnr_current:.2f})")

            if valid_action:
                action_mask[action_index] = 1
            else:
                action_mask[action_index] = 0        # Define a ação dummy (última posição) como válida
        action_mask[-1] = 1

        # ========================
        # Construção final da observação
        # ========================
        # As features de ação agora vêm de mod_features_obs, que tem dimensão:
        # (k_paths * num_bands * modulations_to_consider, 12)
        spectrum_obs_flat = mod_features_obs.flatten().astype(np.float32)
        observation = np.concatenate([
            bit_rate_obs,                         # 1 valor
            source_destination.flatten(),         # 2 valores
            route_lengths.flatten(),              # k_paths valores
            spectrum_obs_flat                     # (k_paths * num_bands * modulations_to_consider * 12) valores
        ], axis=0).astype(np.float32)

        return observation, {'mask': action_mask}



    cpdef simple_decimal_to_array(self, decimal: int, max_values):
        """Decodificação simples sem transformações adicionais - para uso na máscara"""
        array = []
        for max_val in reversed(max_values):
            digit = decimal % max_val
            array.insert(0, digit)
            decimal //= max_val
        return array

    cpdef simple_array_to_decimal(self, array, max_values):
        """Encoding simples sem transformações adicionais - inverso da decodificação simples"""
        decimal = 0
        for i in range(len(array)):
            multiplier = 1
            for j in range(i + 1, len(max_values)):
                multiplier *= max_values[j]
            decimal += array[i] * multiplier
        return decimal

    cpdef decimal_to_array(self, decimal: int, max_values=None):
        if max_values is None:
            # Agora incluindo banda: [k_path, band_idx, mod_idx_relative, global_slot]
            max_values = [self.k_paths, self.num_bands, self.modulations_to_consider, self.total_slots]
        
        array = []
        for max_val in reversed(max_values):
            digit = decimal % max_val
            array.insert(0, digit)
            decimal //= max_val
        
        # Mapear mod_idx_relative para modulation_idx real (índice 2 agora)
        if self.max_modulation_idx > 1:
            allowed_mods = list(range(self.max_modulation_idx, self.max_modulation_idx - (self.modulations_to_consider - 1), -1))
        else:
            allowed_mods = list(range(0, self.modulations_to_consider))
        
        original_mod = array[2]
        array[2] = allowed_mods[array[2]]  # Ajustado para posição 2
        
        # print(f"k:{self.k_shortest_paths[self.current_service.source,self.current_service.destination][array[0]]}, band: {self.bands[array[1]]}, mod: {self.modulations[array[2]]}, slot: {array[3]}")
        return array

    cpdef encoded_decimal_to_array(self, decimal: int, max_values=None):
        if max_values is None:
            # Para multibanda, usar 4 dimensões: [k_path, band, modulation, slot]
            max_values = [self.k_paths, self.num_bands, self.modulations_to_consider, self.total_slots]
        
        array = []
        # Decomposição do número decimal com base nos valores máximos
        for max_val in reversed(max_values):
            digit = decimal % max_val
            array.insert(0, digit)
            decimal //= max_val
        
        # Cria a lista de modulações permitidas
        if self.max_modulation_idx > 1:
            allowed_mods = list(range(self.max_modulation_idx, self.max_modulation_idx - self.modulations_to_consider, -1))
        else:
            allowed_mods = list(reversed(list(range(0, self.modulations_to_consider))))
        
        # Atualiza o dígito de modulação usando o dígito extraído da decomposição
        # Com 4 dimensões: [path, band, modulation, slot] - modulation está no índice 2
        array[2] = allowed_mods[array[2]]
        
        return array
        
        return array



    cpdef tuple[object, float, bint, bint, dict] step(self, int action):
        cdef int route = -1
        cdef int band_idx = -1  # Nova variável para banda
        cdef int modulation_idx = -1
        cdef int initial_slot = -1
        cdef int number_slots = 0

        cdef double osnr = 0.0
        cdef double ase = 0.0
        cdef double nli = 0.0
        cdef double osnr_req = 0.0

        cdef bint truncated = False
        cdef bint terminated
        cdef int disrupted_services = 0

        cdef object modulation = None
        cdef object path = None
        cdef Band band = None  # Nova variável para banda
        cdef list services_to_measure = []
        cdef dict info

        self.current_service.blocked_due_to_resources = False
        self.current_service.blocked_due_to_osnr = False

        if action == (self.action_space.n - 1):
            self.current_service.accepted = False
            self.current_service.blocked_due_to_resources = False
            self.current_service.blocked_due_to_osnr = False
            self.bl_reject += 1
        else:
            # Usar decodificação com slots por banda: [k_path, band_idx, mod_idx_relative, slot_relative]
            slots_per_band = self.bands[0].num_slots
            decoded = self.decimal_to_array(
                action,
                [self.k_paths, self.num_bands, self.modulations_to_consider, slots_per_band]
            )
            route = decoded[0]
            band_idx = decoded[1]
            modulation_idx = decoded[2]  # Já é o índice real da modulação
            slot_relative = decoded[3]
            
            # Converter slot relativo para slot global
            band = self.bands[band_idx]
            initial_slot = band.slot_start + slot_relative
            
            band = self.bands[band_idx]
            modulation = self.modulations[modulation_idx]
            osnr_req = modulation.minimum_osnr + self.margin
            path = self.k_shortest_paths[
                self.current_service.source,
                self.current_service.destination
            ][route]

            number_slots = self.get_number_slots(
                service=self.current_service,
                modulation=modulation
            )
            
            # Validar se o bloco está totalmente dentro da banda
            if not band.contains_slot_range(initial_slot, number_slots):
                raise ValueError(
                    f"Action invalid: slot range [{initial_slot}, {initial_slot + number_slots}) "
                    f"not within band {band.name} range [{band.slot_start}, {band.slot_end})"
                )
            
            if self.is_path_free(path=path, initial_slot=initial_slot, number_slots=number_slots):
                print(f"[DEBUG] step: ✓ Caminho está livre para alocação")
                self.current_service.path = path
                self.current_service.initial_slot = initial_slot
                self.current_service.number_slots = number_slots
                self.current_service.current_band = band  # Definir banda atual
                
                # Calcular frequência central usando a banda
                self.current_service.center_frequency = band.center_frequency_hz_from_global(initial_slot, number_slots)
                
                self.current_service.bandwidth = self.frequency_slot_bandwidth * number_slots
                self.current_service.launch_power = self.launch_power

                osnr, ase, nli = calculate_osnr(self, self.current_service)

                if osnr >= osnr_req:
                    self.current_service.accepted = True
                    self.current_service.OSNR = osnr
                    self.current_service.ASE = ase
                    self.current_service.NLI = nli
                    self.current_service.current_modulation = modulation
                    self.spectrum_efficiency_metric += modulation.spectral_efficiency
                    self.episode_modulation_histogram[modulation.spectral_efficiency] += 1
                    self._provision_path(path, initial_slot, number_slots)

                    if self.bit_rate_selection == "discrete":
                        self.slots_provisioned_histogram[number_slots] += 1

                    self._add_release(self.current_service)
                else:
                    self.bl_osnr += 1
            else:
                self.current_service.accepted = False
                self.current_service.blocked_due_to_resources = True
                self.bl_resource += 1

        if self.measure_disruptions and self.current_service.accepted:
            services_to_measure = []
            for link in self.current_service.path.links:
                # Usar cache em vez de NetworkX
                node1_id = self.topo_cache.get_node_id(link.node1)
                node2_id = self.topo_cache.get_node_id(link.node2)
                link_index = self.topo_cache.get_edge_index(node1_id, node2_id)
                
                for service_in_link in self.topo_cache.edge_running_services[link_index]:
                    if (service_in_link not in services_to_measure
                            and service_in_link not in self.disrupted_services_list):
                        services_to_measure.append(service_in_link)

            for svc in services_to_measure:
                osnr_svc, ase_svc, nli_svc = calculate_osnr(self, svc)
                if osnr_svc < svc.current_modulation.minimum_osnr:
                    disrupted_services += 1
                    if svc not in self.disrupted_services_list:
                        self.disrupted_services += 1
                        self.episode_disrupted_services += 1
                        self.disrupted_services_list.append(svc)

        if not self.current_service.accepted:
            if action == (self.action_space.n - 1):
                self.actions_taken[self.k_paths, self.total_slots] += 1  # Usar total_slots
            else:
                self.actions_taken[self.k_paths, self.total_slots] += 1  # Usar total_slots

            self.current_service.path = None
            self.current_service.initial_slot = -1
            self.current_service.number_slots = 0
            self.current_service.OSNR = 0.0
            self.current_service.ASE = 0.0
            self.current_service.NLI = 0.0
            self.current_service.current_band = None  # Limpar banda

        if self.file_stats is not None:
            line = "{},{},{},{},".format(
                self.current_service.service_id,
                self.current_service.source_id,
                self.current_service.destination_id,
                self.current_service.bit_rate,
            )
            if self.current_service.accepted:
                line += "{},{},{},{},{},{},{},{},{}".format(
                    self.current_service.path.k,
                    self.current_service.path.length,
                    self.current_service.current_modulation.spectral_efficiency,
                    self.current_service.current_modulation.minimum_osnr,
                    self.current_service.OSNR,
                    self.current_service.ASE,
                    self.current_service.NLI,
                    disrupted_services,
                    len(self.topology.graph["running_services"]),
                )
            else:
                line += "-1,-1,-1,-1,-1,-1,-1,-1,-1"
            line += "\n"
            self.file_stats.write(line)
            self.file_stats.flush()

        if not action == (self.action_space.n - 1):
            reward = self.reward()
        else:
            reward = -6.0
        info = {
            "episode_services_accepted": self.episode_services_accepted,
            "service_blocking_rate": 0.0,
            "episode_service_blocking_rate": 0.0,
            "bit_rate_blocking_rate": 0.0,
            "episode_bit_rate_blocking_rate": 0.0,
            "disrupted_services": 0.0,
            "episode_disrupted_services": 0.0,
            "osnr": osnr,
            "osnr_req": osnr_req,
            "chosen_path_index": route,
            "chosen_slot": initial_slot,
            "episode_defrag_cicles": self.episode_defrag_cicles,
            "episode_service_realocations": self.episode_service_realocations,
        }

        if self.services_processed > 0:
            info["service_blocking_rate"] = float(
                self.services_processed - self.services_accepted
            ) / self.services_processed

        if self.episode_services_processed > 0:
            info["episode_service_blocking_rate"] = float(
                self.episode_services_processed - self.episode_services_accepted
            ) / self.episode_services_processed
            info["episode_service_blocking_rate"] = (
                float(self.episode_services_processed - self.episode_services_accepted)
            ) / float(self.episode_services_processed)

        if self.bit_rate_requested > 0:
            info["bit_rate_blocking_rate"] = float(
                self.bit_rate_requested - self.bit_rate_provisioned
            ) / self.bit_rate_requested

        if self.episode_bit_rate_requested > 0:
            info["episode_bit_rate_blocking_rate"] = float(
                self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
            ) / self.episode_bit_rate_requested

        if self.disrupted_services > 0 and self.services_accepted > 0:
            info["disrupted_services"] = float(self.disrupted_services) / self.services_accepted

        if self.episode_disrupted_services > 0 and self.episode_services_accepted > 0:
            info["episode_disrupted_services"] = float(
                self.episode_disrupted_services / self.episode_services_accepted
            )

        cdef float spectral_eff
        for current_modulation in self.modulations:
            spectral_eff = current_modulation.spectral_efficiency
            key = "modulation_{}".format(str(spectral_eff))
            if spectral_eff in self.episode_modulation_histogram:
                info[key] = self.episode_modulation_histogram[spectral_eff]
            else:
                info[key] = 0

        self._new_service = False
        self.topology.graph["services"].append(self.current_service)
        self._next_service()

        terminated = (self.episode_services_processed == self.episode_length)
        if terminated:
            info["blocked_due_to_resources"] = self.bl_resource
            info["blocked_due_to_osnr"] = self.bl_osnr
            info["rejected"] = self.bl_reject

        observation, mask = self.observation()
        info.update(mask)

        return (observation, reward, terminated, truncated, info)

    cpdef _next_service(self):
        cdef float at
        cdef float ht, time
        cdef str src, dst,  dst_id
        cdef float bit_rate
        cdef object service
        cdef int src_id
        cdef object service_to_release
        cdef float lambd

        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        
        self.current_time = at

        ht = self.rng.expovariate(1.0 / self.mean_service_holding_time)

        src, src_id, dst, dst_id = self._get_node_pair()
        if self.bit_rate_selection == "continuous":
            bit_rate = self.bit_rate_function()
        else:
            bit_rate = self.bit_rate_function()[0]

        service = Service(
            service_id=self.episode_services_processed,
            source=src,
            source_id=src_id,  
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate
        )
        self.current_service = service
        self._new_service = True

        self.services_processed += 1
        self.episode_services_processed += 1

        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate
        if self.bit_rate_selection == "discrete":
            self.bit_rate_requested_histogram[self.current_service.bit_rate] += 1
            self.episode_bit_rate_requested_histogram[self.current_service.bit_rate] += 1

        while len(self._events) > 0:
            time, _, service_to_release = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
                if self.defragmentation:
                    if self.n_defrag_services == 0 or self.episode_services_processed % self.n_defrag_services == 0:
                        self.defragment(self.n_defrag_services)
            else:
                heapq.heappush(self._events, (time, service_to_release.service_id, service_to_release))
                break 
    
    cpdef void set_load(self, double load=-1.0, float mean_service_holding_time=-1.0):
        if load > 0:
            self.load = load
        if mean_service_holding_time > 0:
            self.mean_service_holding_time = mean_service_holding_time
        if self.load > 0 and self.mean_service_holding_time > 0:
            self.mean_service_inter_arrival_time = 1 / (self.load / self.mean_service_holding_time)
        else:
            raise ValueError("Both load and mean_service_holding_time must be positive values.")
    
    cdef tuple _get_node_pair(self):
        # Usar lista de nós do cache em vez de chamar topology.nodes()
        cdef list nodes = self.topo_cache.node_names
        
        cdef str src = self.rng.choices(nodes, weights=self.node_request_probabilities)[0]
        cdef int src_id = self.topo_cache.get_node_id(src)  # Usar cache

        cdef cnp.ndarray[cnp.float64_t, ndim=1] new_node_probabilities = np.copy(self.node_request_probabilities)
        new_node_probabilities[src_id] = 0.0

        new_node_probabilities /= np.sum(new_node_probabilities)

        cdef str dst = self.rng.choices(nodes, weights=new_node_probabilities)[0]
        cdef str dst_id = str(self.topo_cache.get_node_id(dst))  # Usar cache

        return src, src_id, dst, dst_id

    cpdef double _get_network_compactness(self):
            cdef double sum_slots_paths = 0.0  
            cdef double sum_occupied = 0.0     
            cdef double sum_unused_spectrum_blocks = 0.0  

            cdef list running_services = self.topology.graph["running_services"]

            for service in running_services:
                sum_slots_paths += service.number_slots * service.path.hops

            # Usar cache em vez de topology.edges()
            for edge_idx in range(self.topo_cache.num_edges):
                available_slots = self.topo_cache.available_slots[edge_idx, :]

                initial_indices, values, lengths = rle(available_slots)

                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]

                    sum_occupied += lambda_max - lambda_min

                    internal_idx, internal_values, internal_lengths = rle(
                        available_slots[lambda_min:lambda_max]
                    )
                    sum_unused_spectrum_blocks += np.sum(internal_values)

            if sum_unused_spectrum_blocks > 0:
                cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (
                    self.topo_cache.num_edges / sum_unused_spectrum_blocks
                )
            else:
                cur_spectrum_compactness = 1.0  

            return cur_spectrum_compactness

    cpdef int get_number_slots(self, object service, object modulation):
            cdef double required_slots
            required_slots = service.bit_rate / (modulation.spectral_efficiency * self.channel_width)
            return int(math.ceil(required_slots))


    cpdef public is_path_free(self, path, initial_slot: int, number_slots: int):
        """Versão otimizada usando operações vetorizadas"""
        cdef int end = initial_slot + number_slots
        if end > self.num_spectrum_resources:
            return False
        
        cdef int start = initial_slot 
        if end < self.num_spectrum_resources:
            end += 1
        
        # Usar FastPathOps para extrair índices
        cdef cnp.ndarray[cnp.int32_t, ndim=1] edge_indices = self.fast_ops.extract_edge_indices(path)
        cdef int i, edge_idx
        
        # Verificação vetorizada se todos os slots estão livres
        for i in range(edge_indices.shape[0]):
            edge_idx = edge_indices[i]
            if np.any(self.topo_cache.available_slots[edge_idx, start:end] == 0):
                return False
        return True

    cpdef double reward(self):
        cdef double reward_value = 0.0
        cdef double failed_ratio = (self.episode_services_processed - self.episode_services_accepted) / float(self.episode_services_processed)

        if not self.current_service.accepted:
            return -3.0 * (1.0 + failed_ratio)

        reward_value = 1.0

        cdef double current_se = self.current_service.current_modulation.spectral_efficiency
        reward_value += 0.1 * current_se

        cdef double osnr_margin = self.current_service.OSNR - self.current_service.current_modulation.minimum_osnr
        if osnr_margin > 0:
            reward_value -= 0.1 * (osnr_margin ** 0.5)

        if reward_value > 3.0:
            reward_value = 3.0
        elif reward_value < -3.0:
            reward_value = -3.0

    
    cpdef _provision_path(self, object path, cnp.int64_t initial_slot, int number_slots):
        cdef int start_slot = initial_slot
        cdef int end_slot = start_slot + number_slots
        cdef cnp.ndarray[cnp.int32_t, ndim=1] edge_indices

        if end_slot < self.num_spectrum_resources:
            end_slot += 1
        elif end_slot > self.num_spectrum_resources:
            raise ValueError("End slot is greater than the number of spectrum resources.")
        
        # Usar FastPathOps para operação vetorizada
        edge_indices = self.fast_ops.extract_edge_indices(path)
        self.fast_ops.fast_provision_path(edge_indices, start_slot, end_slot, 
                                         self.current_service.service_id, self.current_service)
        
        # Atualizar spectrum_slots_allocation
        cdef int i, edge_idx
        for i in range(edge_indices.shape[0]):
            edge_idx = edge_indices[i]
            self.spectrum_slots_allocation[edge_idx, start_slot:end_slot] = self.current_service.service_id

        # Adicionar à lista global de running services
        self.topology.graph["running_services"].append(self.current_service)

        # Atualizar propriedades do serviço
        self.current_service.path = path
        self.current_service.initial_slot = initial_slot
        self.current_service.number_slots = number_slots
        self.current_service.center_frequency = self.frequency_start + (
            self.frequency_slot_bandwidth * initial_slot
        ) + (
            self.frequency_slot_bandwidth * (number_slots / 2.0)
        )
        self.current_service.bandwidth = self.frequency_slot_bandwidth * number_slots

        # Atualizar contadores
        self.services_accepted += 1
        self.episode_services_accepted += 1

        self.bit_rate_provisioned += self.current_service.bit_rate
        self.episode_bit_rate_provisioned = <cnp.int64_t>(
            self.episode_bit_rate_provisioned + self.current_service.bit_rate
        )

        if self.bit_rate_selection == "discrete":
            self.slots_provisioned_histogram[self.current_service.number_slots] += 1
            self.bit_rate_provisioned_histogram[self.current_service.bit_rate] += 1
            self.episode_bit_rate_provisioned_histogram[self.current_service.bit_rate] += 1

    cpdef void _add_release(self, Service service):
        cdef double release_time
        release_time = service.arrival_time + service.holding_time
        heapq.heappush(self._events, (release_time, service.service_id, service))
    
    cpdef public _release_path(self, service: Service):
        cdef int start_slot = service.initial_slot
        cdef int end_slot = start_slot + service.number_slots
        cdef cnp.ndarray[cnp.int32_t, ndim=1] edge_indices
        
        if end_slot < self.num_spectrum_resources:
            end_slot += 1
        
        # Usar FastPathOps para operação vetorizada
        edge_indices = self.fast_ops.extract_edge_indices(service.path)
        self.fast_ops.fast_release_path(edge_indices, start_slot, end_slot, service)
        
        # Atualizar spectrum_slots_allocation
        cdef int i, edge_idx
        for i in range(edge_indices.shape[0]):
            edge_idx = edge_indices[i]
            self.spectrum_slots_allocation[edge_idx, start_slot:end_slot] = -1

        # Remover da lista global de running services
        self.topology.graph["running_services"].remove(service)


    cpdef _update_link_stats(self, str node1, str node2):
        cdef double last_update
        cdef double time_diff
        cdef double last_util
        cdef double cur_util
        cdef double utilization
        cdef double cur_external_fragmentation
        cdef double cur_link_compactness
        cdef double external_fragmentation
        cdef double link_compactness
        cdef int used_spectrum_slots
        cdef int max_empty
        cdef int lambda_min
        cdef int lambda_max
        cdef object link
        cdef cnp.ndarray[cnp.int32_t, ndim=1] slot_allocation
        cdef list initial_indices
        cdef list values
        cdef list lengths
        cdef list unused_blocks
        cdef list used_blocks
        cdef double last_external_fragmentation
        cdef double last_compactness
        cdef double sum_1_minus_slot_allocation
        cdef double unused_spectrum_slots
        cdef Py_ssize_t allocation_size
        cdef int[:] slot_allocation_view
        cdef int[:] sliced_slot_allocation
        cdef int last_index

        edge_idx = self.topo_cache.get_edge_index(node1, node2) if isinstance(node1, int) and isinstance(node2, int) else self.topo_cache.get_edge_index(self.topo_cache.get_node_id(str(node1)), self.topo_cache.get_node_id(str(node2)))
        edge_data = self.topo_cache.edge_data[edge_idx]
        last_update = edge_data.get("last_update", 0.0)

        last_external_fragmentation = edge_data.get("external_fragmentation", 0.0)
        last_compactness = edge_data.get("compactness", 0.0)

        time_diff = self.current_time - last_update

        if self.current_time > 0:
            last_util = edge_data.get("utilization", 0.0)

            slot_allocation = self.topo_cache.available_slots[edge_idx, :]

            slot_allocation = <cnp.ndarray[cnp.int32_t, ndim=1]> np.asarray(slot_allocation, dtype=np.int32)
            slot_allocation_view = slot_allocation

            used_spectrum_slots = self.num_spectrum_resources - np.sum(slot_allocation)

            cur_util = <double> used_spectrum_slots / self.num_spectrum_resources

            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            edge_data["utilization"] = utilization

        initial_indices_np, values_np, lengths_np = rle(slot_allocation)

        if len(initial_indices_np) != len(lengths_np):
            raise ValueError("initial_indices and lengths have different lengths")

        initial_indices = initial_indices_np.tolist()
        values = values_np.tolist()
        lengths = lengths_np.tolist()

        unused_blocks = [i for i, x in enumerate(values) if x == 1]
        if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
            max_empty = max([lengths[i] for i in unused_blocks])
        else:
            max_empty = 0

        if np.sum(slot_allocation) > 0:
            total_unused_slots = slot_allocation.shape[0] - int(np.sum(slot_allocation))
            cur_external_fragmentation = 1.0 - (<double> max_empty / <double> total_unused_slots)
        else:
            cur_external_fragmentation = 1.0

        used_blocks = [i for i, x in enumerate(values) if x == 0]

        if isinstance(initial_indices, list) and isinstance(lengths, list):
            if len(used_blocks) > 1:
                valid = True
                for idx in used_blocks:
                    if not isinstance(idx, int):
                        valid = False
                        break
                    if idx < 0 or idx >= len(initial_indices):
                        valid = False
                        break
                if not valid:
                    raise IndexError("Invalid indices in used_blocks")

                last_index = len(used_blocks) - 1
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[last_index]] + lengths[used_blocks[last_index]]

                allocation_size = slot_allocation.shape[0]

                if lambda_min < 0 or lambda_max > allocation_size:
                    raise IndexError("lambda_min ou lambda_max fora dos limites")

                if lambda_min >= lambda_max:
                    raise ValueError("lambda_min >= lambda_max")

                sliced_slot_allocation = slot_allocation_view[lambda_min:lambda_max]
                sliced_slot_allocation_np = np.asarray(sliced_slot_allocation)

                internal_idx_np, internal_values_np, internal_lengths_np = rle(sliced_slot_allocation_np)

                internal_values = internal_values_np.tolist()
                unused_spectrum_slots = <double> np.sum(1 - internal_values_np)

                sum_1_minus_slot_allocation = <double> np.sum(1 - slot_allocation)

                if unused_spectrum_slots > 0 and sum_1_minus_slot_allocation > 0:
                    cur_link_compactness = ((<double> (lambda_max - lambda_min)) / sum_1_minus_slot_allocation) * (1.0 / unused_spectrum_slots)
                else:
                    cur_link_compactness = 1.0
            else:
                cur_link_compactness = 1.0
        else:
            raise TypeError("initial_indices or lengths are not lists/arrays")


        external_fragmentation = ((last_external_fragmentation * last_update) + (cur_external_fragmentation * time_diff)) / self.current_time
        edge_data["external_fragmentation"] = external_fragmentation

        link_compactness = ((last_compactness * last_update) + (cur_link_compactness * time_diff)) / self.current_time
        edge_data["compactness"] = link_compactness

        edge_data["last_update"] = self.current_time

    cpdef cnp.ndarray get_available_slots(self, object path):
        """Versão otimizada usando FastPathOps para extrair índices"""
        cdef cnp.ndarray[cnp.int32_t, ndim=1] edge_indices
        cdef cnp.ndarray[cnp.int32_t, ndim=2] available_slots_matrix
        cdef cnp.ndarray[cnp.int32_t, ndim=1] product
        cdef int i, j, num_edges, num_slots
        
        # Usar FastPathOps para extrair índices rapidamente
        edge_indices = self.fast_ops.extract_edge_indices(path)
        num_edges = edge_indices.shape[0]
        num_slots = self.num_spectrum_resources
        
        # Extrair matriz de available_slots usando indexação vetorizada
        available_slots_matrix = self.topo_cache.available_slots[edge_indices, :]
        
        # Operação AND vetorizada para obter produto
        product = available_slots_matrix[0].copy()
        for i in range(1, num_edges):
            for j in range(num_slots):
                product[j] *= available_slots_matrix[i, j]
        
        return product
    

    cpdef tuple get_available_blocks(self, int path, int slots, j):
        cdef cnp.ndarray available_slots = self.get_available_slots(
            self.k_shortest_paths[
                self.current_service.source, 
                self.current_service.destination
            ][path]
        )
        cdef cnp.ndarray initial_indices, values, lengths

        initial_indices, values, lengths = rle(available_slots)

        cdef cnp.ndarray available_indices_np = np.where(values == 1)[0]
        cdef cnp.ndarray sufficient_indices_np = np.where(lengths >= slots)[0]
        cdef cnp.ndarray final_indices_np = np.intersect1d(available_indices_np, sufficient_indices_np)[:j]

        return initial_indices[final_indices_np], lengths[final_indices_np]


    cpdef list _get_spectrum_slots(self, int path):
        spectrum_route = []
        for link in self.k_shortest_paths[
            self.current_service.source, 
            self.current_service.destination
        ][path].links:
            # Usar cache em vez de NetworkX
            node1_id = self.topo_cache.get_node_id(link.node1)
            node2_id = self.topo_cache.get_node_id(link.node2)
            link_index = self.topo_cache.get_edge_index(node1_id, node2_id)
            spectrum_route.append(self.topo_cache.available_slots[link_index, :])

        return spectrum_route


    cpdef defragment(self, int num_services):
        self.episode_defrag_cicles += 1
        if num_services == 0:
            num_services = 1000000

        cdef int moved = 0
        cdef Service service
        cdef int number_slots, candidate, i, path_length, link_index, start_slot, end_slot
        cdef object path, candidates
        cdef cnp.ndarray available_slots
        cdef tuple node_list

        cdef list active_services = list(self.topology.graph["running_services"])
        
        cdef int old_initial_slot = 0
        cdef double old_center_frequency =0.0
        cdef double old_bandwidth = 0.0

        for service in active_services:

            old_initial_slot = service.initial_slot
            old_center_frequency = service.center_frequency
            old_bandwidth = service.bandwidth
            
            if moved >= num_services:
                break

            path = service.path
            number_slots = service.number_slots
            available_slots = self.get_available_slots(path)
            
            # Mover serviços apenas dentro da própria banda
            if hasattr(service, 'current_band') and service.current_band is not None:
                candidates = self._get_candidates_in_band(available_slots, number_slots, service.current_band)
            else:
                # Fallback para compatibilidade se não tem banda
                candidates = self._get_candidates(available_slots, number_slots, self.total_slots)

            if not candidates:
                continue

            for candidate in candidates:

                if candidate >= service.initial_slot:
                    continue

                start_slot = candidate
                end_slot = start_slot + number_slots
                if end_slot < self.total_slots:  # Usar total_slots
                    end_slot += 1
                elif end_slot > self.total_slots:  # Usar total_slots
                    continue  

                service.initial_slot = start_slot
                
                # Atualizar center_frequency usando banda se disponível
                if hasattr(service, 'current_band') and service.current_band is not None:
                    service.center_frequency = service.current_band.center_frequency_hz_from_global(start_slot, number_slots)
                else:
                    # Fallback para cálculo original
                    service.center_frequency = self.frequency_start + (
                        self.frequency_slot_bandwidth * start_slot
                    ) + (
                        self.frequency_slot_bandwidth * (number_slots / 2.0)
                    )
                service.bandwidth = self.frequency_slot_bandwidth * number_slots
                service.launch_power = self.launch_power

                osnr, ase, nli = calculate_osnr(self, service)
                if osnr < service.current_modulation.minimum_osnr:
                    service.initial_slot = old_initial_slot
                    service.center_frequency = old_center_frequency
                    service.bandwidth = old_bandwidth
                    continue



                node_list = path.get_node_list()
                path_length = len(node_list)

                # Liberar slots antigos usando cache
                for i in range(path_length - 1):
                    node1_id = self.topo_cache.get_node_id(node_list[i])
                    node2_id = self.topo_cache.get_node_id(node_list[i+1])
                    link_index = self.topo_cache.get_edge_index(node1_id, node2_id)
                    
                    self.topology.graph["available_slots"][link_index,
                        old_initial_slot : old_initial_slot + number_slots + 1] = 1
                    self.spectrum_slots_allocation[link_index,
                        old_initial_slot : old_initial_slot + number_slots + 1] = -1
                    if service in self.topo_cache.edge_running_services[link_index]:
                        self.topo_cache.edge_running_services[link_index].remove(service)

                # Alocar novos slots usando cache
                for i in range(path_length - 1):
                    node1_id = self.topo_cache.get_node_id(node_list[i])
                    node2_id = self.topo_cache.get_node_id(node_list[i+1])
                    link_index = self.topo_cache.get_edge_index(node1_id, node2_id)
                    
                    self.topology.graph["available_slots"][link_index, start_slot:end_slot] = 0
                    self.spectrum_slots_allocation[link_index, start_slot:end_slot] = service.service_id
                    self.topo_cache.edge_services[link_index].append(service)
                    self.topo_cache.edge_running_services[link_index].append(service)

                service.OSNR = osnr
                service.ASE = ase
                service.NLI = nli

                moved += 1
                self.episode_service_realocations += 1

                break

        return

    cpdef public close(self):
        return super().close()
