appliance_param = {
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        'houses': [1, 2, 3],
        'channels': [11, 6, 16],
        'train_build': [2, 3],
        'test_build': 1
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        'houses': [1, 2, 3],
        'channels': [5, 9, 7],
        'train_build': [2, 3],
        'test_build': 1
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        'houses': [1, 2, 3],
        'channels': [6, 10, 9],
        'train_build': [2, 3],
        'test_build': 1
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        'houses': [1, 2, 3],
        'channels': [20, 7, 13],
        'train_build': [2, 3],
        'test_build': 1
    },
    'electric_heat': {
        'windowlength': 599,
        'houses': [1, 5, 6],
        'channels': [13, 12, 12],
        'train_build': [5, 6],
        'test_build': 1
    },
    'air_condition': {
        'windowlength': 599,
        'houses': [6, 4, 6],
        'channels': [16, 10, 15],
        'train_build': [6],
        'test_build': 4
    },
    'disposal': {
        'windowlength': 599,
        'houses': [2, 3, 5],
        'channels': [11, 8, 21],
        'train_build': [3, 5],
        'test_build': 2
    },
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        'houses': [1, 2],
        'channels': [10, 8],
        'train_build': [1],
        'test_build': 2,
    },

    'default_param': {
        'on_power_threshold': 20,
        'mean': 500,
        'std': 800,
    }
}

mains_data = {
    "mean": 522,
    "std":  814
    }

multiple_data = {
    'houses': [1, 2, 3],
    'train_build': [2, 3],
    'test_build': 1
}
