using System;

namespace ML.Core.Data.Loader
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
    public sealed class LoadTypeAttribute : Attribute
    {
        public LoadType LoadType;

        /// <summary>Maps member to specific field in text file.</summary>
        /// <param name="fieldIndex">The index of the field in the text file.</param>
        public LoadTypeAttribute(LoadType loadType)
        {
            LoadType = loadType;
        }
    }

    public enum LoadType
    {
        Feature,
        Label
    }
}