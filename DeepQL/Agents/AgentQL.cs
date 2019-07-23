using System;
using DeepQL.Environments;
using DeepQL.ValueFunc;
using Neuro.Tensors;

namespace DeepQL.Agents
{
    public class AgentQL : Agent
    {
        public AgentQL(string name, Env env, ValueFunctionModel valueFuncModel)
            : base(name, env)
        {
            ValueFuncModel = valueFuncModel;
        }

        protected override Tensor GetOptimalAction()
        {
            return ValueFuncModel.GetOptimalAction(LastObservation);
        }

        protected override void OnStep(int step, int globalStep, Tensor action, float reward, Tensor observation, bool done)
        {
            ValueFuncModel.OnStep(step, globalStep, LastObservation, action, reward, observation, done);
        }

        protected override void OnTrain()
        {
            ValueFuncModel.OnTrain();
        }

        protected override void OnReset(Tensor initialState)
        {
            ValueFuncModel.OnReset(initialState);
        }

        protected override void OnEpisodeEnd(int episode)
        {
            ValueFuncModel.OnEpisodeEnd(episode);
        }

        protected override string GetParametersDescription()
        {
            return $"{base.GetParametersDescription()}\n{ValueFuncModel.GetParametersDescription()}";
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
