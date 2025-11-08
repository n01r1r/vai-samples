#ifndef NDARRAY_H
#define NDARRAY_H


#include <array>
#include <vector>
#include <type_traits>



struct IndexSequenceUtils 
{
private:
	template<typename... Seqs>
	struct cat_impl;

	template<typename Seq>
	struct cat_impl<Seq> { using type = Seq; };

	template<std::size_t... I1, std::size_t... I2, typename... Rest>
	struct cat_impl<std::index_sequence<I1...>, std::index_sequence<I2...>, Rest...> 
    { 
        using type = typename cat_impl<std::index_sequence<I1..., I2...>, Rest...>::type; 
    };

	template<typename Seq, std::size_t... Indices>
	struct permute_impl;

	template<std::size_t... Dims, std::size_t... Indices>
	struct permute_impl<std::index_sequence<Dims...>, Indices...> 
	{
		static constexpr std::size_t dims[] = { Dims... };
		using type = std::index_sequence<dims[Indices]...>;
	};

    template<typename Seq>
    struct to_array_impl;

    template<std::size_t... Ns>
    struct to_array_impl<std::index_sequence<Ns...>>
    {
        static constexpr std::array<std::size_t, sizeof...(Ns)> value = { Ns... };
    };

public:
	template<typename... Seqs>
	using cat = typename cat_impl<Seqs...>::type;

	template<typename Seq, std::size_t... Indices>
	using permute = typename permute_impl<Seq, Indices...>::type;

    template<typename Seq>
    static constexpr auto to_array() 
    {
        return to_array_impl<Seq>::value;
    }
};


struct Uniqueness
{
private:
	template <std::size_t V, std::size_t... Rest>
	static constexpr bool first_repeats = ((V == Rest) || ...);

	template <std::size_t...>
	struct impl;

	template <>
	struct impl<> : std::true_type {};

	template <std::size_t First, std::size_t... Rest>
	struct impl<First, Rest...> : std::bool_constant<!first_repeats<First, Rest...>&& impl<Rest...>::value> {};

public:
	template <std::size_t... Values>
	static constexpr bool check = impl<Values...>::value;
};


struct NDArrayBuilder 
{
private:
	template <typename T, std::size_t... Dims>
	struct Impl;

	template <typename T, std::size_t First, std::size_t... Rest>
	struct Impl<T, First, Rest...> { using type = std::array<typename Impl<T, Rest...>::type, First>; };

	template <typename T>
	struct Impl<T> { using type = T; };

public:
	template <typename T, std::size_t... Dims>
	using build = typename Impl<T, Dims...>::type;
};

template <typename T, std::size_t... Dims>
using NDArray = NDArrayBuilder::build<T, Dims...>;

template <std::size_t... Dims>
using FNDArray = NDArray<float, Dims...>;


struct NDArrayTraits 
{
private:
	template <typename T>
	struct ValueTypeImpl { using type = T; };

	template <typename T, std::size_t N>
	struct ValueTypeImpl<std::array<T, N>> { using type = typename ValueTypeImpl<T>::type; };

	template <typename T>
	struct ShapeImpl { using type = std::index_sequence<>; };

	template <typename T, std::size_t N>
	struct ShapeImpl<std::array<T, N>> { using type = IndexSequenceUtils::cat< std::index_sequence<N>, typename ShapeImpl<T>::type >; };

	template <typename T>
	struct RankImpl : std::integral_constant<std::size_t, 0> {};

	template <typename T, std::size_t N>
	struct RankImpl<std::array<T, N>> : std::integral_constant<std::size_t, 1 + RankImpl<T>::value> {};

	template <typename T>
	struct NumElementsImpl : std::integral_constant<std::size_t, 1> {};

	template <typename T, std::size_t N>
	struct NumElementsImpl<std::array<T, N>> : std::integral_constant<std::size_t, N* NumElementsImpl<T>::value> {};

public:
	template <typename T>
	using valueTypeOf = typename ValueTypeImpl<T>::type;

	template <typename T>
	using shapeOf = typename ShapeImpl<T>::type;

	template <typename T>
	static constexpr std::size_t rankOf = RankImpl<T>::value;

	template <typename T>
	static constexpr std::size_t numElementsOf = NumElementsImpl<T>::value;
};


template <typename T, std::size_t... Dims>
constexpr auto NDArrayFrom(std::index_sequence<Dims...> dims)
{
	return NDArray<T, Dims...>{};
}


template <typename T, typename V>
constexpr void fill(T& arrOrEle, const V& fillValue)
{
	if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>)
		arrOrEle = static_cast<T>(fillValue);
	else
		for (auto& elem : arrOrEle)
			fill(elem, fillValue);
}


template<typename Arr, typename... Idx>
constexpr decltype(auto) getRef(Arr& arr, std::size_t idx, Idx... rest)
{
	if constexpr (sizeof...(rest) == 0)
		return arr[idx];
	else
		return getRef(arr[idx], rest...);
}


template<std::size_t... Perm, typename SrcArr, typename DstArr, std::size_t... Shape, std::size_t... Slots>
constexpr void permuteNDArrayImpl(
	const SrcArr& srcArr,
	DstArr& dstArr,
	std::index_sequence<Shape...>, 
	std::index_sequence<Slots...>)
{
	constexpr std::size_t Rank = sizeof...(Shape);
	constexpr std::size_t srcShape[Rank] = { Shape... };

	std::array<std::size_t, Rank> indices = {};

	for (bool done=false; !done; )
	{
		getRef(dstArr, indices[Perm]...) = getRef(srcArr, indices[Slots]...);

		for (std::size_t dim=Rank; dim-- > 0; )
		{
			if (++indices[dim] < srcShape[dim])
				break; 

			if (dim == 0)
				done = true; 
			else
				indices[dim] = 0;
		}
	}
}


template<std::size_t... Perm, typename Arr>
constexpr auto permute(const Arr& srcArr)
{
	static_assert(Uniqueness::check<Perm...>, "Permutation indices must be unique.");
	static_assert(sizeof...(Perm) == NDArrayTraits::rankOf<Arr>, "Number of permutation indices must match the rank of the array.");
	static_assert((... && (Perm < NDArrayTraits::rankOf<Arr>)), "Permutation indices must be less than the rank of the array.");

	using PermutedShape = IndexSequenceUtils::permute<NDArrayTraits::shapeOf<Arr>, Perm...>;
	using ValueType = NDArrayTraits::valueTypeOf<Arr>;

	auto dstArr = NDArrayFrom<ValueType>(PermutedShape{});
	permuteNDArrayImpl<Perm...>(srcArr, dstArr, NDArrayTraits::shapeOf<Arr>{}, std::make_index_sequence<sizeof...(Perm)>{});
	return dstArr;
}


template<std::size_t... DstShape, typename Arr>
auto& reshapedRef(const Arr& srcArr)
{
	using ValueType = NDArrayTraits::valueTypeOf<Arr>;
	using DstArr = NDArray<ValueType, DstShape...>;
	static_assert(NDArrayTraits::numElementsOf<Arr> == NDArrayTraits::numElementsOf<DstArr>);
	return *((DstArr*)&srcArr);     // This prevents the function from being constexpr
}


template<typename Arr, typename... Sizes>
constexpr bool isShapeOf(const Arr& arr, Sizes... sizes)
{
	std::array<std::size_t, sizeof...(Sizes)> dstShape = { static_cast<std::size_t>(sizes)... };
	auto srcShape = IndexSequenceUtils::to_array<NDArrayTraits::shapeOf<Arr>>();
	return srcShape == dstShape;
}


template <typename Arr>
auto toVector(const Arr& arr) 
{
    using ElemT = NDArrayTraits::valueTypeOf<Arr>;
    std::vector<ElemT> out(NDArrayTraits::numElementsOf<Arr>);
    std::memcpy(out.data(), &arr, out.size() * sizeof(ElemT));
    return out;
}


#endif // NDARRAY_H