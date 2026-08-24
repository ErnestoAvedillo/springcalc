
OPEN_GROUND = 1
CLOSED_GROUND = 2
OPEN_UNGROUND = 3
CLOSED_UNGROUND = 4

COMPRESSION_SPRING_END_TYPES = {
    1: 'open_ground',
    2: 'closed_ground',
    3: 'open_unground',
    4: 'closed_unground'
}

FUNCTIONS_FREE_LENGTH = {
    OPEN_GROUND
}

COMPRESSION_END_FACTORS = {
    COMPRESSION_SPRING_END_TYPES[CLOSED_GROUND]: {'inactive_coils': 2, 'length_factor': 1.5, 'active_coils_offset': 1.5},
    COMPRESSION_SPRING_END_TYPES[OPEN_GROUND]: {'inactive_coils': 0, 'length_factor': 0, 'active_coils_offset': 0},
    COMPRESSION_SPRING_END_TYPES[CLOSED_UNGROUND]: {'inactive_coils': 2, 'length_factor': 3, 'active_coils_offset': 1.5},
    COMPRESSION_SPRING_END_TYPES[OPEN_UNGROUND]: {'inactive_coils': 0, 'length_factor': 1, 'active_coils_offset': 0},
}

EXTENSION_SPRING_END_TYPES = {
    1: 'german_double_full_loop_centered',
    2: 'german_double_full_loop_offset',
    3: 'german_single_loop_centered',
    4: 'german_single_full_loop_centered',
    5: 'german_single_full_loop_offset',
    6: 'special_loop',
    7: 'english_hook'
}

WAHL_FACTOR_CONSTANTS = {
    'red': [0,4],
    'orange': [4,6],
    'green': [6,12]
}
