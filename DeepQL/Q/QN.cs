using System.Collections.Generic;
using DeepQL.Q;
using Neuro.Tensors;

namespace DeepQL.Q
{
    // This is Q function approximation using neural network accepting state and action as an input and returning single Q value
    public class QN : QFunc
    {
        public QN(Shape inputShape, int numberOfActions) : base(inputShape, numberOfActions) {}

        public override int GetBestAction(Tensor state)
        {
            //go through all actions and find the one with max Q value at given state
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
