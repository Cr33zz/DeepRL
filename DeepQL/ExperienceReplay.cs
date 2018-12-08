using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepQL
{
    public class ExperienceReplay
    {
        public ExperienceReplay(int capacity)
        {
            Capacity = capacity;
        }

        public void Push(Experience trans)
        {
            if (trans == null)
                throw new ArgumentNullException();

            if (Memory.Count < Capacity)
                Memory.Add(trans);
            else
                Memory[NextIndex] = trans;

            NextIndex = (NextIndex + 1) % Capacity;
        }

        public List<Experience> Sample(int batchSize)
        {
            var sample = new List<Experience>();
            for (int i = 0; i < batchSize; ++i)
                sample.Add(Memory[GlobalRandom.Rng.Next(StorageSize)]);
            return sample;
        }

        public readonly int Capacity;
        public int StorageSize => Memory.Count;

        private List<Experience> Memory = new List<Experience>();
        private int NextIndex;
    }
}
