using System.Collections.Generic;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DQN : ValueFunctionModel
    {
        public DQN(Shape inputShape, int numberOfActions, double learningRate, double discountFactor)
            : base(inputShape, numberOfActions, learningRate, discountFactor)
        {}

        public override double GetOptimalAction(Tensor state)
        {
            throw new System.NotImplementedException();
        }

        public override void OnTransition(Tensor state, int action, double reward, Tensor nextState)
        {
            throw new System.NotImplementedException();
        }

        protected override void Train(List<Transition> transitions)
        {
            throw new System.NotImplementedException();
        }
    }
}
