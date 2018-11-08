using System.Collections.Generic;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public abstract class ValueFunctionModel
    {
        protected ValueFunctionModel(Shape inputShape, int numberOfActions, double learningRate, double discountFactor)
        {
            LearningRate = learningRate;
            DiscountFactor = discountFactor;
            NumberOfActions = numberOfActions;
        }

        public abstract double GetOptimalAction(Tensor state);

        public abstract void OnTransition(Tensor state, int action, double reward, Tensor nextState);
        
        protected abstract void Train(List<Transition> transitions);

        protected void Train(Transition trans)
        {
            Train(new List<Transition>() { trans });
        }

        protected readonly double LearningRate;
        protected readonly double DiscountFactor;
        public readonly int NumberOfActions;
    }
}
