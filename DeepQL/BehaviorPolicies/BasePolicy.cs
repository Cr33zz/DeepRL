using Neuro.Tensors;

namespace DeepQL.BehaviorPolicies
{
    public abstract class BasePolicy
    {
        public BasePolicy(int numberOfActions)
        {
            NumberOfActions = numberOfActions;
        }

        public abstract int GetNextAction(Tensor state);

        public readonly int NumberOfActions;
    }
}
