using System;
using System.Collections.Generic;

namespace DeepQL.MemoryReplays
{
    public class ExperienceReplay : BaseExperienceReplay
    {
        public ExperienceReplay(int capacity)
            : base(capacity)
        {
        }

        public override void Push(Experience trans)
        {
            if (trans == null)
                throw new ArgumentNullException();

            if (Memory.Count < Capacity)
                Memory.Add(trans);
            else
                Memory[NextIndex] = trans;

            NextIndex = (NextIndex + 1) % Capacity;
        }

        public override List<Experience> Sample(int batchSize)
        {
            var sample = new List<Experience>();
            for (int i = 0; i < batchSize; ++i)
                sample.Add(Memory[GlobalRandom.Rng.Next(GetSize())]);
            return sample;
        }

        public override int GetSize()
        {
            return Memory.Count;
        }

        private readonly List<Experience> Memory = new List<Experience>();
        private int NextIndex;
    }
}
