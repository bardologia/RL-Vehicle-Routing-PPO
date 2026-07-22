import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from core.state import State, StateHandler
from core.environment import Environment
from core.graph import Graph
from core.mask import MaskContext
from core.model import Policy, Action


@dataclass
class InferenceStep:
    step_number    : int
    state          : State
    action         : Optional[Action] = None
    action_info    : Optional[Dict] = None
    reward_info    : Optional[Dict] = None
    cost           : int = 0
    num_routes     : int = 0
    num_unassigned : int = 0
    
    def to_dict(self) -> Dict:
        return {
            "step_number"    : self.step_number,
            "cost"           : self.cost,
            "num_routes"     : self.num_routes,
            "num_unassigned" : self.num_unassigned,
            "operator"       : self.action.operator if self.action else None,
            "vehicle_index"  : self.action.vehicle_index if self.action else None,
            "job_index"      : self.action.job_index if self.action else None,
        }


@dataclass
class InferenceResult:
    steps          : List[InferenceStep] = field(default_factory=list)
    initial_state  : Optional[State] = None
    final_state    : Optional[State] = None
    stopped_reason : str = ""
    total_steps    : int = 0
    
    def get_initial_cost(self) -> int:
        return self.initial_state.cost if self.initial_state else 0
    
    def get_final_cost(self) -> int:
        return self.final_state.cost if self.final_state else 0
    
    def get_cost_improvement(self) -> int:
        return self.get_initial_cost() - self.get_final_cost()
    
    def get_cost_improvement_percentage(self) -> float:
        initial = self.get_initial_cost()
        if initial == 0:
            return 0.0
        return (self.get_cost_improvement() / initial) * 100.0
    
    def summary(self) -> Dict:
        return {
            "total_steps"          : self.total_steps,
            "stopped_reason"       : self.stopped_reason,
            "initial_cost"         : self.get_initial_cost(),
            "final_cost"           : self.get_final_cost(),
            "cost_improvement"     : self.get_cost_improvement(),
            "cost_improvement_pct" : self.get_cost_improvement_percentage(),
            "initial_routes"       : self.initial_state.num_routes if self.initial_state else 0,
            "final_routes"         : self.final_state.num_routes if self.final_state else 0,
            "initial_unassigned"   : self.initial_state.num_unassigned if self.initial_state else 0,
            "final_unassigned"     : self.final_state.num_unassigned if self.final_state else 0,
        }
    
    def get_trajectory(self) -> List[Dict]:
        return [step.to_dict() for step in self.steps]


class ModelInference:
    def __init__(
        self,
        model       : Policy,
        environment : Environment,
        max_steps   : int = 10,
        device      : str = "cuda",
        verbose     : bool = True
    ):

        self.model = model
        self.environment = environment
        self.max_steps = max_steps
        self.device = device
        self.verbose = verbose
        
        self.model.eval()
        self.model.to(self.device)
        
        self.graph_builder = Graph(self.environment.config)
        self.mask_builder  = MaskContext()
    
    def _create_graph(self, state: State):

        graph = self.graph_builder.build(
            self.environment.jobs,
            self.environment.vehicles,
            state.to_dict()
        )
        return graph
    
    def _get_mask_info(self, state: State) -> Dict:
      
        original_state = self.environment.current_state
        self.environment.current_state = state
        mask_info = self.mask_builder.build(self.environment)
        self.environment.current_state = original_state
        
        return mask_info
    
    def _apply_action(self, state: State, action: Action) -> Tuple[State, State]:
 
        original_state = self.environment.current_state
        self.environment.current_state = state
        old_state, new_state = self.environment.apply_action(action)
        self.environment.current_state = original_state
        
        return old_state, new_state
    
    def run(self, initial_state: State):
        result               = InferenceResult()
        result.initial_state = initial_state.copy()
        
        current_state = initial_state.copy()
        step = 0
        
        initial_step = InferenceStep(
            step_number    = 0,
            state          = current_state.copy(),
            action         = None,
            cost           = current_state.cost,
            num_routes     = current_state.num_routes,
            num_unassigned = current_state.num_unassigned
        )
        
        result.steps.append(initial_step)
        
        while step < self.max_steps:
            step += 1
            
            graph = self._create_graph(current_state)
            graph = graph.to(self.device)
            mask_info = self._get_mask_info(current_state)
            
            with torch.no_grad():
                action_result = self.model.act(graph, mask_info=mask_info)
            
            action = action_result["action"]
            
            if action.operator == 2:
                result.stopped_reason = "model_do_nothing"
                break
            
            old_state, new_state = self._apply_action(current_state, action)
            
            reward_info = None
           
            rewards, costs = self.environment.step(
                old_state, new_state, action.operator
            )
            reward_info = {**rewards, **costs}
            
            operator_names = {0: "INSERT", 1: "REMOVE", 2: "DO_NOTHING", 3: "REOPTIMIZE"}
            operator_name = operator_names.get(action.operator, f"UNKNOWN({action.operator})")
            
            vehicle_id = self.environment.vehicles[action.vehicle_index]["id"]
            job_id = self.environment.jobs[action.job_index]["id"]
            

            inference_step = InferenceStep(
                step_number=step,
                state=new_state.copy(),
                action=action,
                action_info={
                    "operator": action.operator,
                    "operator_name": operator_name,
                    "vehicle_index": action.vehicle_index,
                    "job_index": action.job_index,
                    "vehicle_id": vehicle_id,
                    "job_id": job_id,
                },
                reward_info=reward_info,
                cost=new_state.cost,
                num_routes=new_state.num_routes,
                num_unassigned=new_state.num_unassigned
            )
            result.steps.append(inference_step)
            
            current_state = new_state
        
        if step >= self.max_steps:
            result.stopped_reason = "max_steps_reached"
        
        result.final_state = current_state.copy()
        result.total_steps = step
        
        return result
    