using DeepQL.Spaces;
using Neuro.Tensors;
using System;

namespace DeepQL.Environments
{
    public abstract class Env : IDisposable
    {
        protected Env(Space actionSpace, Space observationSpace)
        {
            ActionSpace = actionSpace;
            ObservationSpace = observationSpace;
        }

        // Returns true when end state has been reached
        public abstract bool Step(Tensor action, out Tensor observation, out float reward);
        public abstract Tensor Reset();
        public abstract byte[] Render(bool toRgbArray = false);

        public virtual void Seed(int seed = 0) { Rng = seed > 0 ? new Random(seed) : new Random(); }
        public virtual void Dispose() { }

        public bool Step(int action, out Tensor observation, out float reward)
        {
            return Step(new Tensor(new float[] {action}, new Shape(1)), out observation, out reward);
        }

        public Space ActionSpace { get; protected set; }
        public Space ObservationSpace { get; protected set; }
        public Tensor LastAction { get; protected set; }

        // Observation can different from the internal state
        protected virtual Tensor GetObservation() { return State.Clone(); }

        protected Tensor State;
        protected Random Rng = new Random();
    }
}
