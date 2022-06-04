using System;
using System.Collections.Generic;
using System.Linq;
using AutoDiff;
using ML.Core.Data;
using ML.Core.Transform;
using ML.Utilty;
using Numpy;

namespace ML.Core.Models
{
    public class Perceptron<T> : Model<T>
        where T : DataView
    {
        /// <summary>
        ///     Y should be [Batch, One Hot]
        ///     [
        ///     [0,0,1],
        ///     [0,1,0]
        ///     ]
        /// </summary>
        /// <param name="classes"></param>
        public Perceptron(int classes)
        {
            Transformer = new LinearFirstorder();
            Classes = classes;
        }

        public override Transformer Transformer { get; set; }
        public int Classes { set; get; }
        public override string Description { get; }

        public Dictionary<int, NDarray> WeightsDict { set; get; }
        public Dictionary<int, Variable[]> VariablesDict { set; get; }

        internal void InitialVariables()
        {
            VariablesDict = Variables
                .Select((v, i) => (i / Classes, v))
                .GroupBy(p => p.Item1)
                .ToDictionary(
                    p => p.Key,
                    p => p.Select(a => a.v).ToArray());
        }

        public override Term[] CallGraph(NDarray x)
        {
            var lablesPredict = VariablesDict
                .ToDictionary(
                    p => p.Key,
                    p =>
                    {
                        var feature = Transformer.Call(x);
                        return term.matmul(p.Value, feature);
                    });

            throw new NotImplementedException();
        }

        public override NDarray Call(NDarray x)
        {
            var feature = Transformer.Call(x);
            var y_pred = nn.sigmoid(np.matmul(feature, Weights.T));
            return sign(y_pred);
        }

        private NDarray sign(NDarray inputDarray)
        {
            return np.expand_dims(np.argmax(inputDarray, -1), -1);
        }
    }
}