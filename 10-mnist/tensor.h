#ifndef TENSOR_H
#define TENSOR_H


#include "jsonParser.h"
#include "vulkanApp.h"
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>


class BufferPool
{
    // vk::Device device;
    std::unordered_map<
        VkBufferUsageFlags, 
        std::multimap<size_t, std::pair<vk::Buffer, VkMemoryPropertyFlags>> > bufferPool;

public:
    static BufferPool& get()
    {
        static BufferPool instance;
        return instance;
    }

    vk::Buffer requestBuffer(
        vk::Device device,
        uint32_t usageFlags, 
        uint32_t reqMemProps,
        size_t minSize,
        size_t maxSize = size_t(-1)
    )
    {
        auto& subPool = bufferPool[usageFlags];

        for (auto it = subPool.lower_bound(minSize);
             it != subPool.end() && it->first <= maxSize;
             ++it)
        {
            const auto& [buffer, memProps] = it->second;
            if ((memProps & reqMemProps) == reqMemProps)
            {
                vk::Buffer result = std::move(it->second.first);
                subPool.erase(it);
                return result;
            }
        }

        // Create new buffer
        return device.createBuffer({
            .size = minSize,
            .usage = usageFlags,
            .reqMemProps = reqMemProps
        });
    }

    void returnBuffer(vk::Buffer buffer)
    {
        if (buffer) 
        {
            bufferPool[buffer.usage()].emplace(
                buffer.size(), 
                std::make_pair(std::move(buffer), buffer.memoryProperties())
            );
        }
    }
};


class TensorImpl
{
    friend class Tensor;
    std::vector<uint32_t> shape;
    std::vector<float> data; 
    vk::Buffer _buffer; 

public:
    ~TensorImpl()
    {
        BufferPool::get().returnBuffer(_buffer); // temporary code
    }

};


class Tensor 
{
    std::shared_ptr<TensorImpl> impl;

public:
    template <typename... Ts>
    Tensor(Ts... dims) : impl(std::make_shared<TensorImpl>())
    {
        static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
        impl->shape = {static_cast<uint32_t>(dims)...};
    }

    Tensor(const std::vector<uint32_t>& shape) : impl(std::make_shared<TensorImpl>())
    {
        impl->shape = shape;
    }

    Tensor(const JsonParserRef& json) : impl(std::make_shared<TensorImpl>())
    {
        set(json.parseNDArray(impl->shape));
    }
    
    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    Tensor& set(const std::vector<float>& data) &
    {
        _ASSERT(numElements() == data.size());
        impl->data = data;
        return *this;
    }

    Tensor&& set(const std::vector<float>& data) &&
    {
        _ASSERT(numElements() == data.size());
        impl->data = data;
        return std::move(*this);
    }

    Tensor& set(std::vector<float>&& data) &
    {
        _ASSERT(numElements() == data.size());
        impl->data = std::move(data);
        return *this;
    }

    Tensor&& set(std::vector<float>&& data) &&
    {
        _ASSERT(numElements() == data.size());
        impl->data = std::move(data);
        return std::move(*this);
    }

    template <typename... Ts>
    Tensor& reshape(Ts... dims)
    {
        static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
        _ASSERT(numElements() == (static_cast<size_t>(dims) * ...));
        impl->shape = {static_cast<uint32_t>(dims)...};
        return *this;
    }

    template <typename... Ts>
    Tensor& permute(Ts... Perms);

    const std::vector<uint32_t>& shape() const
    {
        static const std::vector<uint32_t> emptyShape{};
        return impl ? impl->shape : emptyShape;
    }

    template <typename... Ts>
    bool isShapeOf(Ts... dims) const
    {
        static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
        if (!impl || impl->shape.size() != sizeof...(Ts))
            return false;

        const uint32_t dimArray[] = { uint32_t(dims)... };

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return ((dimArray[Is] == impl->shape[Is] || dimArray[Is] == uint32_t(-1)) && ...);
        }(std::index_sequence_for<Ts...>{});
    }

    size_t numElements() const
    {
        if (!impl || impl->shape.empty()) return 0;
        size_t sz = 1; for (uint32_t dim : impl->shape) sz *= dim;
        return sz;
    }

    bool validShape() const
    {
        if (!impl || impl->shape.empty()) return false;
        for (uint32_t dim : impl->shape) if (dim == 0) return false;
        return true;
    }

    bool hasData() const
    {
        return impl && !impl->data.empty();
    }

    float* data()
    {
        if (!impl || impl->data.empty()) return nullptr;
        return impl->data.data();
    }

    void clearData()
    {
        _ASSERT(impl);
        impl->data.clear();
    }

    bool isBufferBound() const
    {
        return impl && impl->_buffer;
    }

    void bindBuffer(vk::Buffer buf)
    {
        _ASSERT(impl);
        impl->_buffer = buf;
    }

    void unbindBuffer()
    {
        _ASSERT(impl);
        impl->_buffer = vk::Buffer();
    }

    vk::Buffer buffer() const
    {
        _ASSERT(impl && impl->_buffer);
        return impl->_buffer;
    }
};


template <typename... Ts>
inline Tensor& Tensor::permute(Ts... Perms)
{
    static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
    _ASSERT(impl && !impl->data.empty());
    
    const uint32_t rank = impl->shape.size();
    _ASSERT(sizeof...(Perms) == rank);
    _ASSERT(((Perms < rank) && ...)); // Ensure all Perms are valid indices

    const uint32_t permutation[] = { static_cast<uint32_t>(Perms)... };

    for (size_t i = 0; i < rank; ++i)
        for (size_t j = i + 1; j < rank; ++j)
            _ASSERT(permutation[i] != permutation[j]);

    std::vector<uint32_t> dstShape(impl->shape.size());
    for (size_t i = 0; i < impl->shape.size(); ++i)
        dstShape[i] = impl->shape[permutation[i]];

    std::vector<size_t> srcStrides(rank);
    srcStrides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i)
        srcStrides[i] = srcStrides[i + 1] * impl->shape[i + 1];

    std::vector<size_t> dstStrides(rank);
    dstStrides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i)
        dstStrides[i] = dstStrides[i + 1] * dstShape[i + 1];

    std::vector<float> dstData(impl->data.size());
    std::vector<uint32_t> srcIndices(rank);
    std::vector<uint32_t> dstIndices(rank);
    for (size_t linear = 0; linear < impl->data.size(); ++linear)
    {
        // linear → multi-dim
        size_t remaining = linear;
        for (uint32_t i = 0; i < rank; ++i)
        {
            srcIndices[i] = remaining / srcStrides[i];
            remaining %= srcStrides[i];
        }

        // permute index
        for (uint32_t i = 0; i < rank; ++i)
            dstIndices[i] = srcIndices[permutation[i]];

        // multi-dim → linear (permuted)
        size_t dstLinear = 0;
        for (uint32_t i = 0; i < rank; ++i)
            dstLinear += dstIndices[i] * dstStrides[i];

        dstData[dstLinear] = impl->data[linear];
    }
    
    impl->shape = std::move(dstShape);
    impl->data = std::move(dstData);
    return *this;
}

#endif // TENSOR_H