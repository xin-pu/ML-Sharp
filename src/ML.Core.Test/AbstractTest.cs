using Xunit.Abstractions;

namespace ML.Core.Test
{
    public abstract class AbstractTest
    {
        internal readonly ITestOutputHelper _testOutputHelper;

        protected AbstractTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        internal void print(object obj)
        {
            _testOutputHelper.WriteLine(obj.ToString());
        }
    }
}