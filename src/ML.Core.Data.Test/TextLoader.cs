using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Data.Test
{
    public class TextLoader
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public TextLoader(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }


        internal void print(object obj)
        {
            _testOutputHelper.WriteLine(obj.ToString());
        }

        [Fact]
        public void Test1()
        {
            print(1);
        }
    }
}