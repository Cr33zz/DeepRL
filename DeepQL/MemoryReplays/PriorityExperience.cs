using Neuro.Tensors;

namespace DeepQL.MemoryReplays
{
    public class PriorityExperience : Experience
    {
        public PriorityExperience(Tensor state, Tensor action, float reward, Tensor nextState, bool done)
            : base(state, action, reward, nextState, done)
        { }

        public int TreeIndex;
        public float Priority;
    }
}
