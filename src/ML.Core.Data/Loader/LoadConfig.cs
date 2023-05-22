using CommunityToolkit.Mvvm.ComponentModel;

namespace ML.Core.Data.Loader
{
    public class LoadConfig : ObservableObject
    {
        private bool _hasHead = true;
        private string _splitChar = ";:,\t";
        private double _splitRatio = 0.8;
        private bool _splitTrainAndVal = true;

        public bool SplitTrainAndVal
        {
            get => _splitTrainAndVal;
            set => SetProperty(ref _splitTrainAndVal, value);
        }

        public double SplitRatio
        {
            get => _splitRatio;
            set => SetProperty(ref _splitRatio, value);
        }

        public string SplitChar
        {
            get => _splitChar;
            set => SetProperty(ref _splitChar, value);
        }

        public bool HasHead
        {
            get => _hasHead;
            set => SetProperty(ref _hasHead, value);
        }
    }
}