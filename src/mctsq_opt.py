"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

mctsq_opt: 
> Computes the potential rewards, the load identifiers, and the theoretical 
  optimum T-OPT ignoring plant dynamics.

Abbreviations:
   SNG: Synthetic natural gas
   EUA: European emission allowances
   CHP: Combined heat and power plant
   CH4: Methane
   H2: Hydrogen
   O2: Oxygen
   CO2: Carbon dioxide
   H2O_DE: Water vapor (steam)
   LHV: Lower heating value
   EEG: German Renewable Energy Act (Erneuerbare-Energien-Gesetz)

----------------------------------------------------------------------------------------------------
"""

import numpy as np
import math

from src.mctsq_config_env import EnvConfiguration

def calculate_optimum(el_price_data: np.array, gas_price_data: np.array, eua_price_data: np.array, data_name: str, stats_names):
    """
    Computes the theoretical maximum revenue for the Power-to-Gas process, assuming no operational constraints.
    :param el_price_data: Electricity market data
    :param gas_price_data: Gas market data
    :param eua_price_data: EUA market data
    :param data_name: Identifier for the dataset
    :param stats_names: List of statistical variable names for tracking results
    :return stats_dict_opt: Dictionary containing process and economic data for the theoretical optimal scenario
    """

    EnvConfig = EnvConfiguration()

    meth_stats = EnvConfig.meth_stats_load      # Methanation process data for partial and full load

    stats_dict_opt = {}                         # Dictionary to store computed results
    stats = np.zeros((len(el_price_data), len(stats_names)))

    # Scenario 3 includes CHP (Combined Heat and Power) revenue calculation
    b_s3 = 1 if EnvConfig.scenario == 3 else 0  

    rew_l = [0,1]       # First entry of the list is dedicated to partial load, the second to full load
    cum_rew = 0         # Cumulative reward

    for t in range(len(el_price_data)):     # Loop over the electricity price data
        t_day = int(math.floor(t / 24))     # Convert hourly index to daily index
        if t_day == len(gas_price_data): t_day -= 1
        for l in range(len(rew_l)):   # Iterate over partial and full load scenarios               
             # Compute revenues and costs for different operating conditions

            # Gas proceeds (Scenario 1+2):          If Scenario == 3: self.gas_price_h[0] = 0
            ch4_volumeflow = meth_stats['Meth_CH4_flow'][l+1] * EnvConfig.convert_mol_to_Nm3            # in [Nm³/s]
            h2_res_volumeflow = meth_stats['Meth_H2_res_flow'][l+1] * EnvConfig.convert_mol_to_Nm3      # in [Nm³/s]
            Q_ch4 = ch4_volumeflow * EnvConfig.H_u_CH4 * 1000                                           # Thermal power of methane in [kW]
            Q_h2_res = h2_res_volumeflow * EnvConfig.H_u_H2 * 1000                                      # Thermal power of residual hydrogen in [kW]
            ch4_revenues = (Q_ch4 + Q_h2_res) * gas_price_data[t_day]                                   # SNG revenues in [ct/h]

            # CHP revenues (Scenario 3):               If Scenario == 3: self.b_s3 = 1 else self.b_s3 = 0
            power_chp = Q_ch4 * EnvConfig.eta_CHP * b_s3                        # Electrical power of the CHP in [kW]
            Q_chp = Q_ch4 * (1 - EnvConfig.eta_CHP) * b_s3                      # Thermal power of the produced steam in the CHP in [kW]
            chp_revenues = power_chp * EnvConfig.eeg_el_price                   # EEG tender revenues in [ct/h]

            # Steam revenues (Scenario 1+2+3):          If Scenario != 3: self.Q_chp = 0
            Q_steam = meth_stats['Meth_H2O_flow'][l+1] * (EnvConfig.dt_water * EnvConfig.cp_water + EnvConfig.h_H2O_evap) / 3600    # Thermal power of the produced steam in the methanation plant in [kW]
            steam_revenues = (Q_steam + Q_chp) * EnvConfig.heat_price                                                               # in [ct/h]

            # Oxygen revenues (Scenario 1+2+3):
            h2_volumeflow = meth_stats['Meth_H2_flow'][l+1] * EnvConfig.convert_mol_to_Nm3          # in [Nm³/s]
            o2_volumeflow = 1 / 2 * h2_volumeflow * 3600                                            # in [Nm³/h] = [Nm³/s * 3600 s/h]
            o2_revenues = o2_volumeflow * EnvConfig.o2_price                                        # Oxygen revenues in [ct/h]

            # EUA revenues (Scenario 1+2):              If Scenario == 3: self.eua_price_h[0] = 0
            Meth_CO2_mass_flow = meth_stats['Meth_CH4_flow'][l+1] * EnvConfig.Molar_mass_CO2 / 1000     # Consumed CO2 mass flow in [kg/s]
            eua_revenues = Meth_CO2_mass_flow / 1000 * 3600 * eua_price_data[t_day] * 100               # EUA revenues in ct/h = kg/s * t/1000kg * 3600 s/h * €/t * 100 ct/€

            # Linear regression model for LHV efficiency of a 6 MW electrolyzer
            # Costs for electricity:
            elec_costs_heating = meth_stats['Meth_el_heating'][l+1] / 1000 * el_price_data[t]       # Electricity costs for methanation heating in [ct/h]
            load_elec = h2_volumeflow / EnvConfig.max_h2_volumeflow                                 # Electrolyzer load
            if load_elec < EnvConfig.min_load_electrolyzer:
                eta_electrolyzer = 0.02
            else:
                eta_electrolyzer = (0.598 - 0.325 * load_elec ** 2 + 0.218 * load_elec ** 3 +
                                    0.01 * load_elec ** (-1) - 1.68 * 10 ** (-3) * load_elec ** (-2) +
                                    2.51 * 10 ** (-5) * load_elec ** (-3))
            elec_costs_electrolyzer = h2_volumeflow * EnvConfig.H_u_H2 * 1000 / eta_electrolyzer * el_price_data[t] # Electricity costs for water electrolysis in [ct/h]
            elec_costs = elec_costs_heating + elec_costs_electrolyzer

            # Costs for water consumption:
            water_elec = meth_stats['Meth_H2_flow'][l+1] * EnvConfig.Molar_mass_H2O / 1000 * 3600                       # Water demand of the electrolyzer in [kg/h] (1 mol water is consumed for producing 1 mol H2)
            water_costs = (meth_stats['Meth_H2O_flow'][l+1] + water_elec) / EnvConfig.rho_water * EnvConfig.water_price # Water costs in [ct/h] = [kg/h / (kg/m³) * ct/m³]

            rew_l[l] = (ch4_revenues + chp_revenues + steam_revenues + eua_revenues +
                        o2_revenues - elec_costs - water_costs)  # in ct/h
        
        # Select the best option (partial or full load)
        tmp = max(rew_l)
        index = rew_l.index(tmp)
        rew = max(rew_l)

        # Store results
        stats[t, 0] = t
        stats[t, 1] = el_price_data[t]
        stats[t, 2] = gas_price_data[t_day]
        stats[t, 3] = eua_price_data[t_day]

        if rew > 0:
            stats[t, 4:20] = [meth_stats['Meth_State'][index + 1], 
                              meth_stats['Meth_Action'][index + 1],
                              meth_stats['Meth_Hot_Cold'][index + 1],
                              meth_stats['Meth_T_cat'][index + 1],
                              meth_stats['Meth_H2_flow'][index + 1],
                              meth_stats['Meth_CH4_flow'][index + 1],
                              meth_stats['Meth_H2O_flow'][index + 1],
                              meth_stats['Meth_el_heating'][index + 1],
                              ch4_revenues, steam_revenues, o2_revenues, eua_revenues, chp_revenues,
                              -elec_costs_heating, -elec_costs_electrolyzer, -water_costs
            ]
            stats[t, 23] = index
            cum_rew += rew
        else:
            stats[t, 4:20] = [meth_stats['Meth_State'][0], 
                              meth_stats['Meth_Action'][0],
                              meth_stats['Meth_Hot_Cold'][0],
                              meth_stats['Meth_T_cat'][0],
                              meth_stats['Meth_H2_flow'][0],
                              meth_stats['Meth_CH4_flow'][0],
                              meth_stats['Meth_H2O_flow'][0],
                              meth_stats['Meth_el_heating'][0]] + [0] * 8
            stats[t, 23] = -1

        # Update reward statistics
        stats[t, 20] = rew
        stats[t, 21] = cum_rew

    # Store computed values in dictionary
    for m in range(len(stats_names)):
        stats_dict_opt[stats_names[m]] = stats[:, m]

    # Print cumulative reward (only if not 'reward_Level')
    if data_name != "reward_Level":
        max_pot_cum_rew = stats_dict_opt['Meth_cum_reward_stats'][-EnvConfig.price_ahead]
        print("    > ", data_name, ": Cumulative reward - theoretical optimum T-OPT = ", round(max_pot_cum_rew,2))
    else:
        max_pot_cum_rew = stats_dict_opt['Meth_cum_reward_stats'][0]

    return stats_dict_opt
