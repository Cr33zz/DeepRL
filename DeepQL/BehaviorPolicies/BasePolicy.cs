using DeepQL.Spaces;
using Neuro.Tensors;

namespace DeepQL.BehaviorPolicies
{
    public abstract class BasePolicy
    {
        public BasePolicy(Space actionSpate)
        {
            ActionSpace = actionSpate;
        }

        public abstract Tensor GetNextAction(Tensor state);

        public readonly Space ActionSpace;
    }
}
