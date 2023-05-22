using System.Collections.Generic;
using System.Linq;

namespace ML.ReinforceTest
{
    /// <summary>
    ///     ¶Ä²©»ú
    /// </summary>
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
            return Help.RandomSelect(Para.Keys.ToArray(), Para.Values.ToArray());
        }

        public override string ToString()
        {
            return Name;
        }
    }
}