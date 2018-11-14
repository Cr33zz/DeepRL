using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Tensors;

namespace DeepQL.Tests
{
    [TestClass]
    public class ReplayMemoryTests
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Push_Null()
        {
            ReplayMemory memory = new ReplayMemory(1);
            memory.Push(null);
        }

        [TestMethod]
        public void Push_Valid_Transition()
        {
            ReplayMemory memory = new ReplayMemory(1);
            Assert.AreEqual(memory.StorageSize, 0);
            AddRandomTransitions(memory, 1);
            Assert.AreEqual(memory.StorageSize, 1);
        }

        [TestMethod]
        public void Push_More_Than_Capacity()
        {
            ReplayMemory memory = new ReplayMemory(10);
            AddRandomTransitions(memory, 15);
            Assert.AreEqual(memory.StorageSize, 10);
        }

        [TestMethod]
        public void Sample_10_Out_Of_5()
        {
            ReplayMemory memory = new ReplayMemory(5);
            AddRandomTransitions(memory, 5);
            var sample = memory.Sample(10);
            Assert.AreEqual(sample.Count, 10);
        }

        [TestMethod]
        public void Sample_2_Out_Of_5()
        {
            ReplayMemory memory = new ReplayMemory(5);
            AddRandomTransitions(memory, 5);
            var sample = memory.Sample(2);
            Assert.AreEqual(sample.Count, 2);
        }

        private void AddRandomTransitions(ReplayMemory memory, int count)
        {
            Random random = new Random();

            for (int i = 0; i < count; ++i)
                memory.Push(new Transition(new Tensor(new Shape(1)), new Tensor(new Shape(1)), random.NextDouble(), new Tensor(new Shape(1)), false));
        }
    }
}
