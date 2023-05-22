using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Random;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace ML.ReinforceTest
{
    public class BanditsTest : AbstractTest
    {
        private readonly Bandits Bandit1 = new Bandits("A", new Dictionary<int, double> {[1] = 0.4, [0] = 0.6});
        private readonly Bandits Bandit2 = new Bandits("B", new Dictionary<int, double> {[1] = 0.2, [0] = 0.8});

        public BanditsTest(ITestOutputHelper testOutputHelper) : base(testOutputHelper)
        {
        }


        [Fact]
        public void TestBandits()
        {
            var bandits = new Bandits("A", new Dictionary<int, double> {[1] = 0.2, [2] = 0.1, [3] = 0.7});

            var p = 0;
            Enumerable.Range(0, 100).ToList().ForEach(i =>
            {
                var g = bandits.Guess();
                if (g == 3)
                    p++;
            });

            print(p / 100.0);
        }


        [Fact]
        public void GreedyTest()
        {
            var all = new List<Bandits> {Bandit1, Bandit2};
            var ep = 1E-2;
            var tryCount = 5000;
            var decay = 100;
            var Q = new Dictionary<Bandits, double> {[Bandit1] = 0, [Bandit2] = 0};
            var C = new Dictionary<Bandits, int> {[Bandit1] = 0, [Bandit2] = 0};
            var randomSource = SystemRandomSource.Default;

            var r = 0;
            Enumerable.Range(0, tryCount).ToList().ForEach(i =>
            {
                if (i > decay)
                    ep = 1 / Math.Sqrt(i);
                Bandits k;
                if (randomSource.NextDouble() < ep)
                {
                    k = all[randomSource.Next(all.Count)];
                }
                else
                {
                    var max = Q.Values.Max();
                    var suit = Q.Count(q => Math.Abs(q.Value - max) < 0.1);
                    k = suit > 1
                        ? Q.Where(q => Math.Abs(q.Value - max) < 0.1).ToList()[randomSource.Next(suit)].Key
                        : Q.First(q => Math.Abs(q.Value - max) < 0.1).Key;
                }

                var v = k.Guess();

                r += v;
                Q[k] = 1.0 * (Q[k] * C[k] + v) / (C[k] + 1);
                C[k] += 1;
                var rate = 1.0 * r / (i + 1);
                print($"{i}\t{rate}");
            });
        }


        [Fact]
        public void SoftmaxTest()
        {
            var all = new List<Bandits> {Bandit1, Bandit2};
            var temp = 1E-13;
            var tryCount = 5000;
            var decay = 100;
            var Q = all.ToDictionary(p => p, p => 0.0);
            var C = all.ToDictionary(p => p, p => 0);
            var r = 0;

            Enumerable.Range(0, tryCount).ToList().ForEach(i =>
            {
                var p = Help.GetBoltzmann(Q, temp);
                var k = Help.RandomSelect(all.ToArray(), p);

                var v = k.Guess();

                r += v;
                Q[k] = 1.0 * (Q[k] * C[k] + v) / (C[k] + 1);
                C[k] += 1;
                var rate = 1.0 * r / (i + 1);
                print($"{i}\t{rate}");
            });
        }
    }


    public class Help
    {
        public static T RandomSelect<T>(T[] objects, double[] per)
        {
            var index = np.asarray(Enumerable.Range(0, objects.Length).ToArray());
            var indexselect = np.random.choice(index, new[] {1}, p: np.asarray(per));
            var finalIndex = indexselect.GetData<int>()[0];
            return objects[finalIndex];
        }

        public static double[] GetBoltzmann(Dictionary<Bandits, double> q, double temp = 0.1)
        {
            var e_q = q.Select(a => Math.Exp(a.Value / temp)).ToList();
            e_q = e_q.Select(i => double.IsInfinity(i) ? double.MaxValue : i).ToList();
            var sum = e_q.Sum();
            var per = e_q.Select(i => i / sum).ToArray();
            if (per.Any(i => double.IsNaN(i))) ;
            return per;
        }
    }
}