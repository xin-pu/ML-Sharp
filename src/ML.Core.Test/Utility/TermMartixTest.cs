using AutoDiff;
using ML.Utility;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test.Utility
{
    public class TermMartixTest : AbstractTest
    {
        public TermMartixTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void Create()
        {
            var tm = new TermMatrix(2, 2)
            {
                [0, 0] = new Constant(1),
                [0, 1] = new Constant(2)
            };

            print(tm);
        }
    }
}