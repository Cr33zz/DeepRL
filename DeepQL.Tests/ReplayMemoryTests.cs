using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using DeepQL.MemoryReplays;
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
            ExperienceReplay memory = new ExperienceReplay(1);
            memory.Push(null);
        }

        [TestMethod]
        public void Push_Valid_Transition()
        {
            ExperienceReplay memory = new ExperienceReplay(1);
            Assert.AreEqual(memory.GetSize(), 0);
            AddRandomTransitions(memory, 1);
            Assert.AreEqual(memory.GetSize(), 1);
        }

        [TestMethod]
        public void Push_More_Than_Capacity()
        {
            ExperienceReplay memory = new ExperienceReplay(10);
            AddRandomTransitions(memory, 15);
            Assert.AreEqual(memory.GetSize(), 10);
        }

        [TestMethod]
        public void Sample_10_Out_Of_5()
        {
            ExperienceReplay memory = new ExperienceReplay(5);
            AddRandomTransitions(memory, 5);
            var sample = memory.Sample(10);
            Assert.AreEqual(sample.Count, 10);
        }

        [TestMethod]
        public void Sample_2_Out_Of_5()
        {
            ExperienceReplay memory = new ExperienceReplay(5);
            AddRandomTransitions(memory, 5);
            var sample = memory.Sample(2);
            Assert.AreEqual(sample.Count, 2);
        }

        private void AddRandomTransitions(ExperienceReplay memory, int count)
        {
            Random random = new Random();

            for (int i = 0; i < count; ++i)
                memory.Push(new Experience(new Tensor(new Shape(1)), new Tensor(new Shape(1)), (float)random.NextDouble(), new Tensor(new Shape(1)), false));
        }
    }
}
