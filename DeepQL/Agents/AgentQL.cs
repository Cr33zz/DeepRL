using System;
using DeepQL.Environments;
using DeepQL.ValueFunc;
using Neuro.Tensors;

namespace DeepQL.Agents
{
    public class AgentQL : Agent
    {
        public AgentQL(Env env, ValueFunctionModel valueFuncModel, bool verbose = false)
            : base(env, verbose)
        {
            ValueFuncModel = valueFuncModel;
        }

        protected override Tensor GetOptimalAction()
        {
            return ValueFuncModel.GetOptimalAction(LastObservation);
        }

        protected override void OnStep(int step, Tensor action, double reward, Tensor observation, bool done)
        {
            ValueFuncModel.OnTransition(LastObservation, action, reward, observation, done);
        }

        protected override void Save()
        {
            throw new NotImplementedException();
        }

        protected override void Load(int episode)
        {
            throw new NotImplementedException();
        }
        private readonly ValueFunctionModel ValueFuncModel;
    }
}
