using DeepQL.MemoryReplays;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DeepQL.Tests
{
    [TestClass]
    public class SumTreeTests
    {
        [TestMethod]
        public void Add()
        {
            var t = new SumTree(6);
            t.Add(new Experience(null, null, 1, null, false), 1);

            Assert.AreEqual(t.GetNodeValue(2), 1);
        }
    }
}
