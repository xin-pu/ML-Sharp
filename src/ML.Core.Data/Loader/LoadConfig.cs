using GalaSoft.MvvmLight;

namespace ML.Core.Data.Loader
{
    public class LoadConfig : ViewModelBase
    {
        private bool _hasHead = true;
        private string _splitChar = ";:,\t";
        private double _splitRatio = 0.8;
        private bool _splitTrainAndVal = true;

        public bool SplitTrainAndVal
        {
            get => _splitTrainAndVal;
            set => Set(ref _splitTrainAndVal, value);
        }

        public double SplitRatio
        {
            get => _splitRatio;
            set => Set(ref _splitRatio, value);
        }

        public string SplitChar
        {
            get => _splitChar;
            set => Set(ref _splitChar, value);
        }

        public bool HasHead
        {
            get => _hasHead;
            set => Set(ref _hasHead, value);
        }
    }
}