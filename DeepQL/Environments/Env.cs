using Neuro.Tensors;

namespace DeepQL.Environments
{
    public abstract class Env
    {
        protected Env(Spaces.Space actionSpace, Spaces.Space observationSpace)
        {
            ActionSpace = actionSpace;
            ObservationSpace = observationSpace;
        }

        // Returns true when end state had been reached
        public abstract bool Step(int action, out Tensor observation, out double reward);
        public abstract Tensor Reset();
        public abstract void Render();
        public abstract void Seed(int seed = 0);

        public readonly Spaces.Space ActionSpace;
        public readonly Spaces.Space ObservationSpace;
    }
}
