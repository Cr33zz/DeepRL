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
        public abstract bool Step(Tensor action, out Tensor observation, out double reward);
        public abstract Tensor Reset();
        public abstract byte[] Render(bool toRgbArray = false);
        public virtual void Seed(int seed = 0) { }
        public virtual void Dispose() { }

        public bool Step(int action, out Tensor observation, out double reward)
        {
            return Step(new Tensor(new double[] {action}, new Shape(1)), out observation, out reward);
        }

        public readonly Space ActionSpace;
        public readonly Space ObservationSpace;
        public Tensor State { get; protected set; }
        public Tensor LastAction { get; protected set; }

    }
}
