using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepQL
{
    public class ReplayMemory
    {
        public ReplayMemory(int capacity)
        {
            Capacity = capacity;
        }

        public void Push(Transition trans)
        {
            if (trans == null)
                throw new ArgumentNullException();

            if (Memory.Count < Capacity)
                Memory.Add(trans);
            else
                Memory[NextIndex] = trans;

            NextIndex = (NextIndex + 1) % Capacity;
        }

        public List<Transition> Sample(int batchSize)
        {
            return new List<Transition>();
        }

        public int StorageSize => Memory.Count;

        private List<Transition> Memory = new List<Transition>();
        private int NextIndex;
        private readonly int Capacity;
    }
}
