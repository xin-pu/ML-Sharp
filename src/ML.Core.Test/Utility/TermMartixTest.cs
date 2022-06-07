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

        [Fact]
        public void TestClone()
        {
            var v1 = new Variable();
            var v2 = new Variable();
            var t = new TermMatrix(2, 1)
            {
                [0, 0] = v1,
                [0, 1] = v2
            };

            var p = t.Power(2);

            var resT = t[0, 0].Evaluate(new[] {v1}, new[] {2.0});
            var resP = p[0, 0].Evaluate(new[] {v1}, new[] {2.0});
            print(resT);
            print(resP);
        }
    }
}