#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "tensor.h"
#include "error.h"
#include <set>
#include <list>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <cstring>  // memcpy


using namespace vk;

class Node;
class NodeSlot;
class Edge;
struct NodeFlow;



class NodeSlot
{
    friend class NeuralNet;
    friend class Edge;

public:
    enum Type { input, output, internal }; ;

private:
    Type _type;
    Node& parent;
    std::list<const Edge*> edges; 

    Tensor value; 
    // bool valueIsConstant = false;
    
public:
    NodeSlot(Type _type, Node* parent) : _type(_type), parent(*parent) {}
    bool isOk()
    {
        if(_type == input)
            return edges.size() == 1;
        else if (_type == output)
            return edges.size() >= 0;
    }

    Type type() const
    {
        return _type;
    }

    Tensor& getValueRef()
    {
        return value;
    }

    const Tensor& getValueRef() const
    {
        return value;
    }
};


class Edge 
{
    struct Hash 
    {
        std::size_t operator()(const Edge& edge) const 
        {
            return std::hash<NodeSlot*>()(&edge._from) 
                ^ (std::hash<NodeSlot*>()(&edge._to) << 1);
        }
    };
    inline static std::unordered_set<Edge, Edge::Hash> edgePool;

    NodeSlot& _from;
    NodeSlot& _to;

public:
    Edge(NodeSlot& _from, NodeSlot& _to) : _from(_from), _to(_to) {}

    static void connect(NodeSlot& from, NodeSlot& to)
    {
        auto [it, inserted] = edgePool.emplace(from, to);
        if (inserted)
        {
            const Edge* pEdge = &(*it);
            from.edges.push_back(pEdge);
            to.edges.push_back(pEdge);
        }
    }

    static void disconnect(NodeSlot& from, NodeSlot& to)
    {
        auto it = edgePool.find({from, to});
        if (it != edgePool.end())
        {
            const Edge* pEdge = &(*it);
            from.edges.remove(pEdge);
            to.edges.remove(pEdge);
            edgePool.erase(it);
        }
    }

    bool operator==(const Edge& other) const 
    {
        return &_from == &other._from && &_to == &other._to;
    }

    NodeSlot& from() const 
    {
        return _from;
    }

    NodeSlot& to() const 
    {
        return _to;
    }
};


class Node
{
    friend class NeuralNet;
    std::string name = "noname";
    std::map<std::string, NodeSlot> slots;

protected:
    void addSlot(const std::string& name, NodeSlot::Type type)
    {
        slots.try_emplace(name, type, this);
    }

public:
    void setName(const std::string& name)
    {
        this->name = name;
    }

    NodeSlot& slot(const std::string& name)
    {
        return slots.at(name);
    }

    Tensor& operator[](const std::string& name)
    {
        return slots.at(name).getValueRef();
    }

    const Tensor& operator[](const std::string& name) const
    {
        return slots.at(name).getValueRef();
    }

    operator NodeFlow();

    virtual void prepare() = 0;
    virtual void run(CommandBuffer cmdBuff) = 0;
    virtual ~Node() = default;
};


class InputNode : public Node
{
    friend class NeuralNet;
public:
    InputNode() 
    {
        addSlot("in0", NodeSlot::input);
        addSlot("out0", NodeSlot::output);
    }

    void prepare() override
    {
        _ASSERT((*this)["in0"].validShape());
        (*this)["out0"] = (*this)["in0"];
    }
    
    void run(CommandBuffer cmdBuff) override
    {
    }   
};


class OutputNode : public Node
{
    friend class NeuralNet;
public:
    OutputNode()
    {
        addSlot("in0", NodeSlot::input);
        addSlot("out0", NodeSlot::output);
    }

    void prepare() override
    {
        _ASSERT((*this)["in0"].validShape());
        (*this)["out0"] = (*this)["in0"];
    }
    
    void run(CommandBuffer cmdBuff) override
    {
    }  
};


class NeuralNet
{
    Device& device;
    
    // std::map<std::string, Node*> nodes;
    std::vector<Node*> sortedNodes;
    std::vector<std::vector<Node*>> linearChains;

    BufferPool& bufferPool = BufferPool::get();
    Buffer uploadBuffer; 
    uint8_t* uploadBufferMappedAddress = nullptr;
    size_t uploadBufferOffset = 0; 
    const size_t uploadBufferSize = 1024 * 1024 * 64; // 64 MB

    std::vector<InputNode> _inputs;
    std::vector<OutputNode> _outputs;

public:
    NeuralNet(Device& device, uint32_t numInputs = 1, uint32_t numOutputs = 1);

    void reset()
    {
    }

    // Node& addNode(const std::string& name, Node&& node)
    // {
    //     node.setName(name);
    //     auto [it, inserted] = nodes.try_emplace(name, std::move(node));
    //     if (!inserted)
    //         throw std::runtime_error("Node with the same name already exists: " + name);
    //     return it->second;
    // }

    InputNode& input(uint32_t index=0)
    {
        _ASSERT(index < _inputs.size());
        return _inputs[index];
    }

    OutputNode& output(uint32_t index=0)
    {
        _ASSERT(index < _outputs.size());
        return _outputs[index];
    }

    void sortNodes(bool buildChains = false);
    void prepare();
    void run();

    /*
        TODO: At the moment, only r-value tensors can return their underlying buffers 
        to the pool once all references are released. 
        Handling for l-values is not yet implemented.
    */
    template <typename... Ts>
    std::vector<Tensor> operator()(Ts&&... tensors)
    {
        constexpr size_t N = sizeof...(Ts);

        _ASSERT(N == _inputs.size());
        static_assert((... && std::is_same_v<Tensor, std::decay_t<Ts>>));

        [&]<std::size_t... Is>(std::index_sequence<Is...>) 
        {
            (..., (_inputs[Is]["in0"] = std::forward<Ts>(tensors)));
        }(std::make_index_sequence<N>{});

        run();
        
        std::vector<Tensor> results(_outputs.size());
        for (uint32_t i = 0; i < _outputs.size(); ++i) 
            results[i] = std::move(_outputs[i]["out0"]);

        return results;
    }
};

inline NeuralNet::NeuralNet(Device& device, uint32_t numInputs, uint32_t numOutputs) 
: device(device)
{
    uploadBuffer = device.createBuffer({
        .size = uploadBufferSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });

    uploadBufferOffset = 0;
    uploadBufferMappedAddress = uploadBuffer.map();
    
    _inputs.resize(numInputs);
    _outputs.resize(numOutputs);
}

inline void NeuralNet::sortNodes(bool buildChains)
{
    std::unordered_map<Node*, std::vector<Node*>> outNodes;
    std::unordered_map<Node*, std::vector<Node*>> inNodes;
    std::unordered_map<Node*, uint32_t> inDegree;
    std::unordered_set<Node*> visited;

    auto dfs = [&](auto& self, Node& node) -> void
    {
        if (!visited.insert(&node).second)
            return;

        for (auto& [name, slot] : node.slots)
        {
            if (slot.type() != NodeSlot::output)
                continue;

            for (const Edge* edge : slot.edges)
            {
                Node& nextNode = edge->to().parent;
                outNodes[&node].push_back(&nextNode);
                inNodes[&nextNode].push_back(&node);
                inDegree[&nextNode]++;
                self(self, nextNode);
            }
        }
    };

    for (Node& inputNode : _inputs)
    {
        inDegree[&inputNode] = 0; // ensure input nodes have in-degree 0
        inNodes[&inputNode] = {}; // ensure input nodes have empty in-nodes
        dfs(dfs, inputNode);
    }

    sortedNodes.clear();

    // 1. Topological sort (Kahn's algorithm)
    std::queue<Node*> q;

    for (const auto& [node, deg] : inDegree)
    {
        if (deg == 0)
            q.push(node);
    }

    while (!q.empty())
    {
        Node* curr = q.front(); q.pop();
        sortedNodes.push_back(curr);

        for (Node* next : outNodes[curr])
        {
            if (--inDegree[next] == 0)
                q.push(next);
        }
    }

    if (sortedNodes.size() != visited.size())
        throw std::runtime_error("Cycle detected in graph!");

    // 2. Extract linear chains in topological order (not used for now)
    if (!buildChains)
        return;

    linearChains.clear();
    visited.clear();

    for (Node* node : sortedNodes)
    {
        if (visited.count(node)) continue;

        std::vector<Node*> chain;
        Node* curr = node;

        while (
            inNodes[curr].size() == 1 &&
            outNodes[inNodes[curr][0]].size() == 1)
        {
            curr = inNodes[curr][0];
        }

        chain.push_back(curr);
        visited.insert(curr);

        while (
            outNodes[curr].size() == 1 &&
            inNodes[outNodes[curr][0]].size() == 1)
        {
            curr = outNodes[curr][0];
            chain.push_back(curr);
            visited.insert(curr);
        }

        linearChains.push_back(std::move(chain));
    }
}


inline void NeuralNet::prepare()
{
}

/*
Assume that tensors are assigned only to input slots that require explicit user input.
*/
inline void NeuralNet::run()
{
    if (sortedNodes.empty())
        sortNodes();

    auto cmdBuffer = device.newCommandBuffer(queue_compute).begin();

    for (Node* node : sortedNodes)
    {
        // - Verify whether each input slot is assigned to a tensor and the tensor has the correct shape.
        // - Assign tensor for output/internal slots
        node->prepare();

        // - Share tensor of output slots with connected input slots
        // - Input slots are classified as two types 
        //      - Connected by some outpuslot       -> share the tensor with the output slot
        //      - Not connected by any output slot  -> data must be feed by user
        for (auto& [name, slot] : node->slots)
        {
            Tensor& tensor = slot.getValueRef();

            if (slot.type() == NodeSlot::input)
            {
                if (!slot.edges.empty())
                {
                    _ASSERT(slot.edges.size() == 1);
                    _ASSERT(!tensor.hasHostData() && tensor.hasDeviceData());
                }
                else
                {
                    if (tensor.hasHostData())
                    {
                        _ASSERT(!tensor.hasDeviceData());

                        size_t byteSize = tensor.numElements() * sizeof(float);
                        
                        Buffer buffer = bufferPool.requestBuffer(
                            device,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            byteSize,
                            // tensor.isConstant() ? byteSize * 1.5f : byteSize * 4.0f
                            byteSize * 1.5f
                        );
                        
                        tensor.bindBuffer(buffer);
                        
                        memcpy(uploadBufferMappedAddress + uploadBufferOffset, tensor.hostData(), byteSize);
    
                        cmdBuffer
                            .copyBuffer(buffer, uploadBuffer(uploadBufferOffset, byteSize))
                            .barrier(
                                (PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_WRITE) 
                                / buffer 
                                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
                            );
    
                        uploadBufferOffset += byteSize;
                        tensor.clearHostData(); 
                    }
                }
            }
            else // if (slot.type() != NodeSlot::input)
            {
                if (!tensor.hasDeviceData())
                {
                    tensor.bindBuffer(bufferPool.requestBuffer(
                        device,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT // TODO: only needed for final output tensors, so we should remove for general case. 
                        , 
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        tensor.numElements() * sizeof(float)
                    ));
                }
                else
                {
                    /*
                    - InputNode - out0 slot
                    - OutputNode - out0 slot
                    - FlattenNode - out0 slot
                    */
                }
                    
                if (slot.type() == NodeSlot::output)
                {
                    for (const Edge* edge : slot.edges)
                        edge->to().getValueRef() = tensor;
                }
            }
        }

        // Record the command buffer for executing the program of the node
        node->run(cmdBuffer);        

        // invlaidate the tensor to return the bound buffer to the pool
        for (auto& [name, slot] : node->slots)
        {
            if (slot.type() == NodeSlot::output && slot.edges.size() == 0)
                continue;

            if (slot.getValueRef().isConstant())
                continue;

            slot.getValueRef() = Tensor(); 
        }
    }

    device.queue() << cmdBuffer.end() << waiting;
    uploadBufferOffset = 0; 
}








struct NodeFlow 
{
    std::string inSlotName;
    Node& node;
    std::string outSlotName;
};

inline Node::operator NodeFlow()
{
    return NodeFlow("in0", *this, "out0");
}

inline NodeFlow operator/(std::string name, Node& node)
{
    return { name, node, "out0" };
}

inline NodeFlow operator/(Node& node, std::string name)
{
    return { "in0", node, name };
}

inline NodeFlow&& operator/(NodeFlow&& other, std::string name)
{
    other.outSlotName = name;
    return std::move(other);
}

inline void operator-(NodeSlot& from, NodeSlot& to)
{
    Edge::connect(from, to);
}

inline NodeFlow&& operator-(NodeFlow&& inflow, NodeFlow&& outflow)
{
    inflow.node.slot(inflow.outSlotName) - outflow.node.slot(outflow.inSlotName);
    return std::move(outflow);
}





struct GroupFlow;
class NodeGroup
{
    std::map<std::string, NodeSlot*> slots;

protected:
    void defineSlot(const std::string& name, NodeSlot& slot)
    {
        slots.try_emplace(name, &slot);
    }

public:
    NodeSlot& slot(const std::string& name)
    {
        return *slots.at(name);
    }

    operator GroupFlow();
};

struct GroupFlow
{
    std::string inSlotName;
    NodeGroup& group;
    std::string outSlotName;
};

inline NodeGroup::operator GroupFlow()
{
    return { "in0", *this, "out0" };
}

inline GroupFlow operator/(std::string name, NodeGroup& gp)
{
    return { name, gp, "out0" };
}

inline GroupFlow operator/(NodeGroup& gp, std::string name)
{
    return { "in0", gp, name };
}

inline GroupFlow&& operator/(GroupFlow&& other, std::string name)
{
    other.outSlotName = name;
    return std::move(other);
}

inline GroupFlow&& operator-(GroupFlow&& inflow, GroupFlow&& outflow)
{
    inflow.group.slot(inflow.outSlotName) - outflow.group.slot(outflow.inSlotName);
    return std::move(outflow);
}

inline GroupFlow&& operator-(NodeFlow&& inflow, GroupFlow&& outflow)
{
    inflow.node.slot(inflow.outSlotName) - outflow.group.slot(outflow.inSlotName);
    return std::move(outflow);
}

inline NodeFlow&& operator-(GroupFlow&& inflow, NodeFlow&& outflow)
{
    inflow.group.slot(inflow.outSlotName) - outflow.node.slot(outflow.inSlotName);
    return std::move(outflow);
}



#endif // NEURAL_NET_H