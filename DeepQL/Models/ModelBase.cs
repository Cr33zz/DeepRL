using System.Collections.Generic;
using Neuro.Tensors;

namespace DeepQL.Models
{
    public abstract class ModelBase
    {
        protected ModelBase(Shape inputShape, int numberOfActions)
        {
            NumberOfActions = numberOfActions;
        }

        public abstract void OnTransition(Tensor state, int action, double reward, Tensor nextState);
        
        protected abstract void Train(List<Transition> transitions);

        protected void Train(Transition trans)
        {
            Train(new List<Transition>() { trans });
        }

        public readonly int NumberOfActions;
    }
}
