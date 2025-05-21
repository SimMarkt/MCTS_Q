# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
# GitHub Repository: https://github.com/SimMarkt/RL_PtG
#
# rl_config_env: 
# > Contains the configuration for the Power-to-Gas (PtG) process environment
# > Converts the data from 'config_env.yaml' into a class object for further processing and usage
# ----------------------------------------------------------------------------------------------------------------

import yaml

class EnvConfiguration:
    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config_env.yaml", "r") as env_file:
            env_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(env_config)
        
        assert self.scenario in [1,2,3], f"Specified business scenario ({self.scenario}) must match one of the three implemented scenarios [1,2,3]!"
        self.train_len_d = None                                 # Total number of days in the training set                    
        
        # Construct file paths for process data depending on the load level (OP1 or OP2):
        assert self.operation in ['OP1', 'OP2'], f"Invalid load level specified - data/config_env.yaml -> operation : {env_config['operation']} must match ['OP1', 'OP2']"
        base_path = self.datafile_path['path'] + self.operation
        for i in range(2, 19):          # For datafile_path2 to datafile_path18
            setattr(self, f'datafile_path{i}', f"{base_path}/{self.datafile_path['datafile'][f'datafile_path{i}']}")

        self.meth_stats_load = self.meth_stats_load[self.operation]                                     # Methanation data for steady-state operation for the different load levels
        self.max_h2_volumeflow = self.convert_mol_to_Nm3 *  self.meth_stats_load['Meth_H2_flow'][2]     # Maximum hydrogen production during water electrolysis

        # Lower and upper bounds for normalizing PtG environment (Should be adjusted to the energy market and process data value ranges)
        # The upper bounds for hydrogen (h2_u_b), methane (ch4_u_b), residual hydrogen (h2_res_u_b), and water (h2o_u_b)
        # correspond to the full_load values from meth_stats_load for the selected load level.
        self.h2_u_b = self.meth_stats_load['Meth_H2_flow'][2]           # Upper bound of hydrogen molar flow rate [mol/s]
        self.ch4_u_b = self.meth_stats_load['Meth_CH4_flow'][2]         # Upper bound of methane molar flow rate [mol/s]
        self.h2_res_u_b = self.meth_stats_load['Meth_H2_res_flow'][2]   # Upper bound of residual product gas hydrogen molar flow rate [mol/s]
        self.h2o_u_b = self.meth_stats_load['Meth_H2O_flow'][2]         # Upper bound of water mass flow rate [kg/h]

        # Variable names for statistics, data storage, and evaluation
        self.stats_names = ['steps_stats', 'el_price_stats', 'gas_price_stats', 'eua_price_stats', 'Meth_State_stats',
                            'Meth_Action_stats', 'Meth_Hot_Cold_stats', 'Meth_T_cat_stats', 'Meth_H2_flow_stats',
                            'Meth_CH4_flow_stats', 'Meth_H2O_flow_stats', 'Meth_el_heating_stats', 'Meth_ch4_revenues_stats',
                            'Meth_steam_revenues_stats', 'Meth_o2_revenues_stats', 'Meth_eua_revenues_stats',
                            'Meth_chp_revenues_stats', 'Meth_elec_costs_heating_stats', 'Meth_elec_costs_electrolyzer_stats',
                            'Meth_water_costs_stats', 'Meth_reward_stats', 'Meth_cum_reward_stats', 'pot_reward_stats', 'part_full_stats']
