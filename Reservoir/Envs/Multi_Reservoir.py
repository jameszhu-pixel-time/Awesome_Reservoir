### we aim to design a multi-reservoir system, providing determinstic transition to the input;
### Start with a simple demp
import numpy as np
from typing import List,Tuple,Dict
class Reservoir():
    def __init__(self,time_length=100,S_Q_type='Linear'):
        self.time_length = time_length
        self.Power = np.zeros(time_length) ##  全过程的离散时间点出力; 万kW
        self.Q_s = np.zeros(time_length)  ##  全过程的离散时间点闸口泄水流量 m^3/s 出库流量
        self.V_end = np.zeros(time_length) ##  全过程的离散时间调度期末电站可用水量 亿m^3（存水+ 上游留下来)
        self.Z_height = np.zeros(time_length)  ##  全过程的离散时间调度期水位 m
        #### constraint 1
        self.S_volumn = np.zeros(time_length)   #表示电站 i 在时段 j 的库容 亿m^3
        self.I_inflow = np.zeros(time_length) ## 电站 i 在时段 j 的入库流量 m^3/s
        self.q_flow = np.zeros(time_length) ## 电站 i 在时段 j 的区间流量 m^3/s
        self.Q_gen = np.zeros(time_length) ## 电站 i 在时段 j 的发电流量 m^3/s
        self.Q_s = np.zeros(time_length)  ## 闸门泄流量 m^3/s
        self.Q_outflow = np.zeros(time_length)  ## 总流量 m^3/s
        #### constraint 2
        self.Z_min = np.zeros(time_length)  ## 期末最低水位 m^3
        self.Z_max = np.zeros(time_length)  ## 期末最高水位 m^3
        #### onstraint 3
        self.ZD_min = np.zeros(time_length)  ## 期末最大降幅 m
        self.ZI_max = np.zeros(time_length)  ## 期末最大升幅 m
        #### constraint 4
        self.Q_outflow_min = np.zeros(time_length)   ## 期末最小流出 m^3
        self.Q_outflow_max = np.zeros(time_length)   ## 期末最大流出 m^3
        #### constraint 5
        self.QD_min = np.zeros(time_length)  ## 流量期末最大降幅 m^3/s
        self.QI_max = np.zeros(time_length)  ## 流量期末最大升幅 m^3/s
        #### constraint 6
        self.P_min = np.zeros(time_length)  ## 期末最小出力 软约束m^3/s
        self.P_max = np.zeros(time_length)  ## 期末最大出力 可违反m^3/s
        #### constraint 7
        self.A_reserve = np.zeros(time_length) ##T/F at different time
        if S_Q_type == 'Linear':
            self.S_Q_type = 'Linear'
        else:
            self.S_Q_type = 'Quadra'
            
        
    def initialize(self,data_list):
        #TODO
        '''
        initialize parameter with given data at time t.
        data_list: dictionary:
        keys: same name as attributes
        
        '''
        
        raise NotImplementedError
    def step(self):
        raise NotImplementedError
        
    
    
class BaseEnv():
    def __init__(self,Reservoir_List=None):
        raise NotImplementedError

    def transition(self,**kwargs):
        '''
        Get Decision Variables, then 
        return the transition state according to
        Enviornment transformation
        '''
        
        raise NotImplementedError
    def step(self,**kwargs):
        '''
        Update component of the Reservoir System
        '''
        raise NotImplementedError


class Simple_Env(BaseEnv):
    def __init__(self,Reservoir_List:List =None):
        
        self.Rs = Reservoir_List ##[Reservoir1,.....]
        self.transmit_time = 10 ##constraint 1
        #time
        self.time_length = Reservoir_List[0].time_length
        ## Env constraints
        self.A_max_reserve = np.zeros(self.time_length)
        raise NotImplementedError
    def transition(self,**kwargs):
        '''
        Get Decision Variables, then 
        return the transition state according to
        Enviornment transformation
        input: actions
        '''
        #we first copy the constraints equations
        #we return changed item's dictionary back
        def volumn_balance_eq(re_item:Reservoir,time:int):
            '''
            get S for next time step
            Logic: (Inflow - outflow)*duration = Change of Storage
            returns:
            {"S_volumn":S_ij}
            '''
            result = {}
            
            re_item.S_volumn[time+1] = re_item.S_volumn[time] + 3600*(re_item.I_inflow-re_item.Q_outflow)*self.transmit_time/(1e8)
            result["S_volumn"] = re_item.S_volumn[time+1]
            return result
        def water_level_ineq(re_item:Reservoir,time:int):
            '''
            available Z_min,Z_max
            notes: use it for action bound to achieve hard constraints
            '''
            result = {"Z_min":re_item.Z_min,"Z_max":re_item.Z_max}
            return result
        def water_vary_ineq(re_item:Reservoir,time:int):
            '''
            available Z_min,Z_max
            notes: use it for action bound to achieve hard constraints
            '''
            
        
        raise NotImplementedError
    def step(self,**kwargs):
        '''
        Update component of the Reservoir System
        '''
        raise NotImplementedError
