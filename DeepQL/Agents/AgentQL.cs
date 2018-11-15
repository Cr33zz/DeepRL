using System;
using DeepQL.Environments;
using DeepQL.ValueFunc;
using Neuro.Tensors;

namespace DeepQL.Agents
{
    public class AgentQL : Agent
    {
        public AgentQL(string name, Env env, ValueFunctionModel valueFuncModel, bool verbose = false)
            : base(name, env, verbose)
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

        public override void Save(string filename)
        {
            ValueFuncModel.SaveState(filename);
        }

        public override void Load(string filename)
        {
            ValueFuncModel.LoadState(filename);
        }

        private readonly ValueFunctionModel ValueFuncModel;
    }
}
