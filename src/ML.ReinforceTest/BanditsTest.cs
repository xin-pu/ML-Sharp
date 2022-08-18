using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Random;
using Xunit;
using Xunit.Abstractions;

namespace ML.ReinforceTest
{
    public class BanditsTest : AbstractTest
    {
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
            var b1 = new Bandits("A", new Dictionary<int, double> {[1] = 0.4, [0] = 0.6});

            var b2 = new Bandits("B", new Dictionary<int, double> {[1] = 0.2, [0] = 0.8});


            var all = new List<Bandits> {b1, b2};
            var ep = 1E-1;
            var tryCount = 5000;
            var Q = new Dictionary<Bandits, double> {[b1] = 0, [b2] = 0};
            var C = new Dictionary<Bandits, int> {[b1] = 0, [b2] = 0};
            var randomSource = SystemRandomSource.Default;

            var r = 0;
            Enumerable.Range(0, tryCount).ToList().ForEach(i =>
            {
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
                print($"{k} {v}");
                r += v;
                Q[k] = 1.0 * (Q[k] * C[k] + v) / (C[k] + 1);
                C[k] += 1;
                //print($"{i}\t{r}");
            });
        }
    }

    public class Bandits
    {
        public Bandits(string name, Dictionary<int, double> para)
        {
            Name = name;
            var sum = para.Values.Sum();
            var dict = new Dictionary<int, double>();
            foreach (var keyValuePair in para) dict[keyValuePair.Key] = keyValuePair.Value / sum;
            Para = dict;
        }

        public string Name { get; set; }
        public Dictionary<int, double> Para { set; get; }

        public int Guess()
        {
            var randomSource = SystemRandomSource.Default;
            var value = randomSource.NextDouble();
            var l = 0.0;
            foreach (var p in Para)
            {
                if (value > l && value < p.Value + l)
                    return p.Key;
                l += p.Value;
            }

            return 0;
        }

        public override string ToString()
        {
            return Name;
        }
    }
}