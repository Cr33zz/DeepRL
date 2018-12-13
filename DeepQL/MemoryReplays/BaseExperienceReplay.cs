using System.Collections.Generic;

namespace DeepQL.MemoryReplays
{
    public abstract class BaseExperienceReplay
    {
        protected BaseExperienceReplay(int capacity)
        {
            Capacity = capacity;
        }

        public abstract void Push(Experience trans);
        public abstract List<Experience> Sample(int batchSize);
        public virtual void Update(List<Experience> samples, List<float> absErrors) {}
        public abstract int GetSize();

        public virtual string GetParametersDescription() { return $"{GetType().Name}: capacity={Capacity}"; }

        public readonly int Capacity;
    }
}
