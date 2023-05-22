using ML.Core.Data.Loader;
using Numpy;

namespace ML.Core.Data.DataStructs
{
    [Serializable]
    public class Winequality : DataView
    {
        [LoadColumn(10)] public double Alcohol;

        [LoadColumn(4)] public double Chlorides;

        [LoadColumn(2)] public double CitricAcid;


        [LoadColumn(7)] public double Density;

        [LoadColumn(0)] public double FifixedAcidityxed;

        [LoadColumn(5)] public double FreeSulfurDioxide;
        [LoadColumn(8)] public double PH;

        [LoadColumn(11)] public double Quality;

        [LoadColumn(3)] public double ResidualSugar;

        [LoadColumn(9)] public double Sulphates;

        [LoadColumn(6)] public double TotalSulfurDioxide;

        [LoadColumn(1)] public double VolatileAcidity;

        /// <summary>
        ///     Cons
        /// </summary>
        public Winequality()
        {
        }


        public override NDarray GetFeatureArray()
        {
            return np.array(FifixedAcidityxed, VolatileAcidity, CitricAcid, ResidualSugar, Chlorides, FreeSulfurDioxide,
                TotalSulfurDioxide, Density, PH, Sulphates, Alcohol);
        }

        public override NDarray GetLabelArray()
        {
            return np.array(Quality);
        }


        public override string ToString()
        {
            return "";
        }
    }
}