using System;
using System.Linq;

namespace DeepQL.MemoryReplays
{
    public class SumTree
    {
        public SumTree(int capacity)
        {
            Capacity = capacity;
            Tree = new float[2 * capacity - 1];
            Memory = new Experience[capacity];
        }

        public void Add(Experience trans, float priority)
        {
            if (trans == null)
                throw new ArgumentNullException();

            int leafTreeIndex = NextDataIndex + Capacity - 1;

            Memory[NextDataIndex] = trans;
            trans.Index = leafTreeIndex;

            Update(leafTreeIndex, priority);

            ++NextDataIndex;

            if (NextDataIndex >= Capacity)
                NextDataIndex = 0;
        }

        public Experience GetLeaf(float v, out float priority)
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

                if (v <= Tree[leftChildIndex])
                {
                    parentIndex = leftChildIndex;
                }
                // there is a number error introduced by multiple substractions thus for v close to total priority we may end up with wrong/null experience selected
                // that's why we have to check right child priority and then correct v
                else if (Tree[rightChildIndex] == 0)
                {
                    parentIndex = leftChildIndex;
                    v = Tree[leftChildIndex];
                }
                else
                {
                    v -= Tree[leftChildIndex];
                    parentIndex = rightChildIndex;
                }
            }

            priority = Tree[leafIndex];
            int dataIndex = leafIndex - Capacity + 1;
            return Memory[dataIndex];
        }

        public void Update(int leafTreeIndex, float priority)
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

        public float GetTotalPriority()
        {
            return Tree[0];
        }

        public float GetMaxPriority(int validElementsCount)
        {
            if (validElementsCount <= 0)
                return 0;
            return Tree.Skip(Capacity - 1).Take(validElementsCount).Max();
        }

        public float GetMinPriority(int validElementsCount)
        {
            if (validElementsCount <= 0)
                return 0;
            return Tree.Skip(Capacity - 1).Take(validElementsCount).Min();
        }

        // this method is used in unit tests
        public float GetNodeValue(int index)
        {
            return Tree[index];
        }

        public readonly int Capacity;

        private readonly float[] Tree;
        private readonly Experience[] Memory;
        private int NextDataIndex;
    }
}
