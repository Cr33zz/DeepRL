using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepQL.MemoryReplays
{
    public class PriorityExperienceReplay
    {
        public PriorityExperienceReplay(int capacity)
        {
            Capacity = capacity;
            Tree = new float[2 * capacity - 1];
            Memory = new Experience[capacity];
        }

        public void Push(Experience trans, float priority)
        {
            if (trans == null)
                throw new ArgumentNullException();

            int leafTreeIndex = NextIndex + Capacity - 1;

            Memory[NextIndex] = trans;

            Update(leafTreeIndex, priority);

            ++NextIndex;

            if (NextIndex >= Capacity)
                NextIndex = 0;
        }

        public List<Experience> Sample(int batchSize)
        {
            var sample = new List<Experience>();
            
            // todo

            return sample;
        }

        private Experience SampleLeaf(float priorityValue)
        {
            int leafIndex = 0;
            int parentIndex = 0;

            while (true)
            {
                int leftChildIndex = 2 * parentIndex + 1;
                int rightChildIndex = leftChildIndex + 1;

                if (leftChildIndex >= Tree.Length)
                {
                    leafIndex = parentIndex;
                    break;
                }
                else
                {
                    if (priorityValue <= Tree[leftChildIndex])
                    {
                        parentIndex = leftChildIndex;
                    }
                    else
                    {
                        priorityValue -= Tree[leftChildIndex];
                        parentIndex = rightChildIndex;
                    }
                }
            }

            int memoryIndex = leafIndex - Capacity + 1;
            return Memory[memoryIndex];
        }

        private void Update(int leafTreeIndex, float priority)
        {
            float priorityChange = priority - Tree[leafTreeIndex];
            Tree[leafTreeIndex] = priority;

            int treeIndex = leafTreeIndex;
            while (treeIndex != 0)
            {
                //parent tree index
                treeIndex = (int)Math.Floor((treeIndex - 1) / 2.0);
                Tree[treeIndex] += priorityChange;
            }
        }

        public readonly int Capacity;

        private float[] Tree;
        private Experience[] Memory;
        private int NextIndex;
    }
}
