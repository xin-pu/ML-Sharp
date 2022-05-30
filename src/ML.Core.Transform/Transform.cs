using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Transform
{
    public abstract class Transform : MvxViewModel
    {
        public string Name => GetType().Name;

        public abstract NDarray Call(NDarray inputNDarray);

        public override string ToString()
        {
            return Name;
        }
    }
}