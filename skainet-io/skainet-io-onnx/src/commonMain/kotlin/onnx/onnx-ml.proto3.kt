@file:OptIn(pbandk.PublicForGeneratedCode::class)

package onnx

@pbandk.Export
public sealed class Version(override val value: Int, override val name: String? = null) : pbandk.Message.Enum {
    override fun equals(other: kotlin.Any?): Boolean = other is Version && other.value == value
    override fun hashCode(): Int = value.hashCode()
    override fun toString(): String = "Version.${name ?: "UNRECOGNIZED"}(value=$value)"

    public object _START_VERSION : Version(0, "_START_VERSION")
    public object IR_VERSION_2017_10_10 : Version(1, "IR_VERSION_2017_10_10")
    public object IR_VERSION_2017_10_30 : Version(2, "IR_VERSION_2017_10_30")
    public object IR_VERSION_2017_11_3 : Version(3, "IR_VERSION_2017_11_3")
    public object IR_VERSION_2019_1_22 : Version(4, "IR_VERSION_2019_1_22")
    public object IR_VERSION_2019_3_18 : Version(5, "IR_VERSION_2019_3_18")
    public object IR_VERSION_2019_9_19 : Version(6, "IR_VERSION_2019_9_19")
    public object IR_VERSION_2020_5_8 : Version(7, "IR_VERSION_2020_5_8")
    public object IR_VERSION_2021_7_30 : Version(8, "IR_VERSION_2021_7_30")
    public object IR_VERSION_2023_5_5 : Version(9, "IR_VERSION_2023_5_5")
    public object IR_VERSION_2024_3_25 : Version(10, "IR_VERSION_2024_3_25")
    public object IR_VERSION_2025_05_12 : Version(11, "IR_VERSION_2025_05_12")
    public object IR_VERSION_2025_08_26 : Version(12, "IR_VERSION_2025_08_26")
    public object IR_VERSION : Version(13, "IR_VERSION")
    public class UNRECOGNIZED(value: Int) : Version(value)

    public companion object : pbandk.Message.Enum.Companion<Version> {
        public val values: List<Version> by lazy { listOf(_START_VERSION, IR_VERSION_2017_10_10, IR_VERSION_2017_10_30, IR_VERSION_2017_11_3, IR_VERSION_2019_1_22, IR_VERSION_2019_3_18, IR_VERSION_2019_9_19, IR_VERSION_2020_5_8, IR_VERSION_2021_7_30, IR_VERSION_2023_5_5, IR_VERSION_2024_3_25, IR_VERSION_2025_05_12, IR_VERSION_2025_08_26, IR_VERSION) }
        override fun fromValue(value: Int): Version = values.firstOrNull { it.value == value } ?: UNRECOGNIZED(value)
        override fun fromName(name: String): Version = values.firstOrNull { it.name == name } ?: throw IllegalArgumentException("No Version with name: $name")
    }
}

@pbandk.Export
public sealed class OperatorStatus(override val value: Int, override val name: String? = null) : pbandk.Message.Enum {
    override fun equals(other: kotlin.Any?): Boolean = other is OperatorStatus && other.value == value
    override fun hashCode(): Int = value.hashCode()
    override fun toString(): String = "OperatorStatus.${name ?: "UNRECOGNIZED"}(value=$value)"

    public object EXPERIMENTAL : OperatorStatus(0, "EXPERIMENTAL")
    public object STABLE : OperatorStatus(1, "STABLE")
    public class UNRECOGNIZED(value: Int) : OperatorStatus(value)

    public companion object : pbandk.Message.Enum.Companion<OperatorStatus> {
        public val values: List<OperatorStatus> by lazy { listOf(EXPERIMENTAL, STABLE) }
        override fun fromValue(value: Int): OperatorStatus = values.firstOrNull { it.value == value } ?: UNRECOGNIZED(value)
        override fun fromName(name: String): OperatorStatus = values.firstOrNull { it.name == name } ?: throw IllegalArgumentException("No OperatorStatus with name: $name")
    }
}

@pbandk.Export
public data class AttributeProto(
    val name: String = "",
    val refAttrName: String = "",
    val docString: String = "",
    val type: AttributeProto.AttributeType = AttributeProto.AttributeType.fromValue(0),
    val f: Float = 0.0F,
    val i: Long = 0L,
    val s: pbandk.ByteArr = pbandk.ByteArr.empty,
    val t: TensorProto? = null,
    val g: GraphProto? = null,
    val sparseTensor: SparseTensorProto? = null,
    val tp: TypeProto? = null,
    val floats: List<Float> = emptyList(),
    val ints: List<Long> = emptyList(),
    val strings: List<pbandk.ByteArr> = emptyList(),
    val tensors: List<TensorProto> = emptyList(),
    val graphs: List<GraphProto> = emptyList(),
    val sparseTensors: List<SparseTensorProto> = emptyList(),
    val typeProtos: List<TypeProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): AttributeProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<AttributeProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<AttributeProto> {
        public val defaultInstance: AttributeProto by lazy { AttributeProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): AttributeProto = AttributeProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<AttributeProto> = pbandk.MessageDescriptor(
            fullName = "AttributeProto",
            messageClass = AttributeProto::class,
            messageCompanion = this,
            fields = buildList(18) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "name",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "name",
                        value = AttributeProto::name
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "f",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Primitive.Float(),
                        jsonName = "f",
                        value = AttributeProto::f
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "i",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                        jsonName = "i",
                        value = AttributeProto::i
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "s",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Primitive.Bytes(),
                        jsonName = "s",
                        value = AttributeProto::s
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "t",
                        number = 5,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorProto.Companion),
                        jsonName = "t",
                        value = AttributeProto::t
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "g",
                        number = 6,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = GraphProto.Companion),
                        jsonName = "g",
                        value = AttributeProto::g
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "floats",
                        number = 7,
                        type = pbandk.FieldDescriptor.Type.Repeated<Float>(valueType = pbandk.FieldDescriptor.Type.Primitive.Float(), packed = true),
                        jsonName = "floats",
                        value = AttributeProto::floats
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "ints",
                        number = 8,
                        type = pbandk.FieldDescriptor.Type.Repeated<Long>(valueType = pbandk.FieldDescriptor.Type.Primitive.Int64(), packed = true),
                        jsonName = "ints",
                        value = AttributeProto::ints
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "strings",
                        number = 9,
                        type = pbandk.FieldDescriptor.Type.Repeated<pbandk.ByteArr>(valueType = pbandk.FieldDescriptor.Type.Primitive.Bytes()),
                        jsonName = "strings",
                        value = AttributeProto::strings
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "tensors",
                        number = 10,
                        type = pbandk.FieldDescriptor.Type.Repeated<TensorProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorProto.Companion)),
                        jsonName = "tensors",
                        value = AttributeProto::tensors
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "graphs",
                        number = 11,
                        type = pbandk.FieldDescriptor.Type.Repeated<GraphProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = GraphProto.Companion)),
                        jsonName = "graphs",
                        value = AttributeProto::graphs
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "doc_string",
                        number = 13,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "docString",
                        value = AttributeProto::docString
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "tp",
                        number = 14,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Companion),
                        jsonName = "tp",
                        value = AttributeProto::tp
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "type_protos",
                        number = 15,
                        type = pbandk.FieldDescriptor.Type.Repeated<TypeProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Companion)),
                        jsonName = "typeProtos",
                        value = AttributeProto::typeProtos
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "type",
                        number = 20,
                        type = pbandk.FieldDescriptor.Type.Enum(enumCompanion = AttributeProto.AttributeType.Companion),
                        jsonName = "type",
                        value = AttributeProto::type
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "ref_attr_name",
                        number = 21,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "refAttrName",
                        value = AttributeProto::refAttrName
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "sparse_tensor",
                        number = 22,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = SparseTensorProto.Companion),
                        jsonName = "sparseTensor",
                        value = AttributeProto::sparseTensor
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "sparse_tensors",
                        number = 23,
                        type = pbandk.FieldDescriptor.Type.Repeated<SparseTensorProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = SparseTensorProto.Companion)),
                        jsonName = "sparseTensors",
                        value = AttributeProto::sparseTensors
                    )
                )
            }
        )
    }

    public sealed class AttributeType(override val value: Int, override val name: String? = null) : pbandk.Message.Enum {
        override fun equals(other: kotlin.Any?): Boolean = other is AttributeProto.AttributeType && other.value == value
        override fun hashCode(): Int = value.hashCode()
        override fun toString(): String = "AttributeProto.AttributeType.${name ?: "UNRECOGNIZED"}(value=$value)"

        public object UNDEFINED : AttributeType(0, "UNDEFINED")
        public object FLOAT : AttributeType(1, "FLOAT")
        public object INT : AttributeType(2, "INT")
        public object STRING : AttributeType(3, "STRING")
        public object TENSOR : AttributeType(4, "TENSOR")
        public object GRAPH : AttributeType(5, "GRAPH")
        public object SPARSE_TENSOR : AttributeType(11, "SPARSE_TENSOR")
        public object TYPE_PROTO : AttributeType(13, "TYPE_PROTO")
        public object FLOATS : AttributeType(6, "FLOATS")
        public object INTS : AttributeType(7, "INTS")
        public object STRINGS : AttributeType(8, "STRINGS")
        public object TENSORS : AttributeType(9, "TENSORS")
        public object GRAPHS : AttributeType(10, "GRAPHS")
        public object SPARSE_TENSORS : AttributeType(12, "SPARSE_TENSORS")
        public object TYPE_PROTOS : AttributeType(14, "TYPE_PROTOS")
        public class UNRECOGNIZED(value: Int) : AttributeType(value)

        public companion object : pbandk.Message.Enum.Companion<AttributeProto.AttributeType> {
            public val values: List<AttributeProto.AttributeType> by lazy { listOf(UNDEFINED, FLOAT, INT, STRING, TENSOR, GRAPH, SPARSE_TENSOR, TYPE_PROTO, FLOATS, INTS, STRINGS, TENSORS, GRAPHS, SPARSE_TENSORS, TYPE_PROTOS) }
            override fun fromValue(value: Int): AttributeProto.AttributeType = values.firstOrNull { it.value == value } ?: UNRECOGNIZED(value)
            override fun fromName(name: String): AttributeProto.AttributeType = values.firstOrNull { it.name == name } ?: throw IllegalArgumentException("No AttributeType with name: $name")
        }
    }
}

@pbandk.Export
public data class ValueInfoProto(
    val name: String = "",
    val type: TypeProto? = null,
    val docString: String = "",
    val metadataProps: List<StringStringEntryProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): ValueInfoProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<ValueInfoProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<ValueInfoProto> {
        public val defaultInstance: ValueInfoProto by lazy { ValueInfoProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): ValueInfoProto = ValueInfoProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<ValueInfoProto> = pbandk.MessageDescriptor(
            fullName = "ValueInfoProto",
            messageClass = ValueInfoProto::class,
            messageCompanion = this,
            fields = buildList(4) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "name",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "name",
                        value = ValueInfoProto::name
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "type",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Companion),
                        jsonName = "type",
                        value = ValueInfoProto::type
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "doc_string",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "docString",
                        value = ValueInfoProto::docString
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "metadata_props",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "metadataProps",
                        value = ValueInfoProto::metadataProps
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class NodeProto(
    val input: List<String> = emptyList(),
    val output: List<String> = emptyList(),
    val name: String = "",
    val opType: String = "",
    val domain: String = "",
    val overload: String = "",
    val attribute: List<AttributeProto> = emptyList(),
    val docString: String = "",
    val metadataProps: List<StringStringEntryProto> = emptyList(),
    val deviceConfigurations: List<NodeDeviceConfigurationProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): NodeProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<NodeProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<NodeProto> {
        public val defaultInstance: NodeProto by lazy { NodeProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): NodeProto = NodeProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<NodeProto> = pbandk.MessageDescriptor(
            fullName = "NodeProto",
            messageClass = NodeProto::class,
            messageCompanion = this,
            fields = buildList(10) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "input",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Repeated<String>(valueType = pbandk.FieldDescriptor.Type.Primitive.String()),
                        jsonName = "input",
                        value = NodeProto::input
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "output",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Repeated<String>(valueType = pbandk.FieldDescriptor.Type.Primitive.String()),
                        jsonName = "output",
                        value = NodeProto::output
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "name",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "name",
                        value = NodeProto::name
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "op_type",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "opType",
                        value = NodeProto::opType
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "attribute",
                        number = 5,
                        type = pbandk.FieldDescriptor.Type.Repeated<AttributeProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = AttributeProto.Companion)),
                        jsonName = "attribute",
                        value = NodeProto::attribute
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "doc_string",
                        number = 6,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "docString",
                        value = NodeProto::docString
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "domain",
                        number = 7,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "domain",
                        value = NodeProto::domain
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "overload",
                        number = 8,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "overload",
                        value = NodeProto::overload
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "metadata_props",
                        number = 9,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "metadataProps",
                        value = NodeProto::metadataProps
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "device_configurations",
                        number = 10,
                        type = pbandk.FieldDescriptor.Type.Repeated<NodeDeviceConfigurationProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = NodeDeviceConfigurationProto.Companion)),
                        jsonName = "deviceConfigurations",
                        value = NodeProto::deviceConfigurations
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class IntIntListEntryProto(
    val key: Long = 0L,
    val value: List<Long> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): IntIntListEntryProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<IntIntListEntryProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<IntIntListEntryProto> {
        public val defaultInstance: IntIntListEntryProto by lazy { IntIntListEntryProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): IntIntListEntryProto = IntIntListEntryProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<IntIntListEntryProto> = pbandk.MessageDescriptor(
            fullName = "IntIntListEntryProto",
            messageClass = IntIntListEntryProto::class,
            messageCompanion = this,
            fields = buildList(2) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "key",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                        jsonName = "key",
                        value = IntIntListEntryProto::key
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "value",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Repeated<Long>(valueType = pbandk.FieldDescriptor.Type.Primitive.Int64(), packed = true),
                        jsonName = "value",
                        value = IntIntListEntryProto::value
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class NodeDeviceConfigurationProto(
    val configurationId: String = "",
    val shardingSpec: List<ShardingSpecProto> = emptyList(),
    val pipelineStage: Int = 0,
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): NodeDeviceConfigurationProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<NodeDeviceConfigurationProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<NodeDeviceConfigurationProto> {
        public val defaultInstance: NodeDeviceConfigurationProto by lazy { NodeDeviceConfigurationProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): NodeDeviceConfigurationProto = NodeDeviceConfigurationProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<NodeDeviceConfigurationProto> = pbandk.MessageDescriptor(
            fullName = "NodeDeviceConfigurationProto",
            messageClass = NodeDeviceConfigurationProto::class,
            messageCompanion = this,
            fields = buildList(3) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "configuration_id",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "configurationId",
                        value = NodeDeviceConfigurationProto::configurationId
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "sharding_spec",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Repeated<ShardingSpecProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = ShardingSpecProto.Companion)),
                        jsonName = "shardingSpec",
                        value = NodeDeviceConfigurationProto::shardingSpec
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "pipeline_stage",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int32(),
                        jsonName = "pipelineStage",
                        value = NodeDeviceConfigurationProto::pipelineStage
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class ShardingSpecProto(
    val tensorName: String = "",
    val device: List<Long> = emptyList(),
    val indexToDeviceGroupMap: List<IntIntListEntryProto> = emptyList(),
    val shardedDim: List<ShardedDimProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): ShardingSpecProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<ShardingSpecProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<ShardingSpecProto> {
        public val defaultInstance: ShardingSpecProto by lazy { ShardingSpecProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): ShardingSpecProto = ShardingSpecProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<ShardingSpecProto> = pbandk.MessageDescriptor(
            fullName = "ShardingSpecProto",
            messageClass = ShardingSpecProto::class,
            messageCompanion = this,
            fields = buildList(4) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "tensor_name",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "tensorName",
                        value = ShardingSpecProto::tensorName
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "device",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Repeated<Long>(valueType = pbandk.FieldDescriptor.Type.Primitive.Int64(), packed = true),
                        jsonName = "device",
                        value = ShardingSpecProto::device
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "index_to_device_group_map",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Repeated<IntIntListEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = IntIntListEntryProto.Companion)),
                        jsonName = "indexToDeviceGroupMap",
                        value = ShardingSpecProto::indexToDeviceGroupMap
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "sharded_dim",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Repeated<ShardedDimProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = ShardedDimProto.Companion)),
                        jsonName = "shardedDim",
                        value = ShardingSpecProto::shardedDim
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class ShardedDimProto(
    val axis: Long = 0L,
    val simpleSharding: List<SimpleShardedDimProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): ShardedDimProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<ShardedDimProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<ShardedDimProto> {
        public val defaultInstance: ShardedDimProto by lazy { ShardedDimProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): ShardedDimProto = ShardedDimProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<ShardedDimProto> = pbandk.MessageDescriptor(
            fullName = "ShardedDimProto",
            messageClass = ShardedDimProto::class,
            messageCompanion = this,
            fields = buildList(2) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "axis",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                        jsonName = "axis",
                        value = ShardedDimProto::axis
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "simple_sharding",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Repeated<SimpleShardedDimProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = SimpleShardedDimProto.Companion)),
                        jsonName = "simpleSharding",
                        value = ShardedDimProto::simpleSharding
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class SimpleShardedDimProto(
    val numShards: Long = 0L,
    val dim: Dim<*>? = null,
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    public sealed class Dim<V>(value: V) : pbandk.Message.OneOf<V>(value) {
        public class DimValue(dimValue: Long = 0L) : Dim<Long>(dimValue)
        public class DimParam(dimParam: String = "") : Dim<String>(dimParam)
    }

    val dimValue: Long?
        get() = (dim as? Dim.DimValue)?.value
    val dimParam: String?
        get() = (dim as? Dim.DimParam)?.value

    override operator fun plus(other: pbandk.Message?): SimpleShardedDimProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<SimpleShardedDimProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<SimpleShardedDimProto> {
        public val defaultInstance: SimpleShardedDimProto by lazy { SimpleShardedDimProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): SimpleShardedDimProto = SimpleShardedDimProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<SimpleShardedDimProto> = pbandk.MessageDescriptor(
            fullName = "SimpleShardedDimProto",
            messageClass = SimpleShardedDimProto::class,
            messageCompanion = this,
            fields = buildList(3) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "dim_value",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int64(hasPresence = true),
                        oneofMember = true,
                        jsonName = "dimValue",
                        value = SimpleShardedDimProto::dimValue
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "dim_param",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(hasPresence = true),
                        oneofMember = true,
                        jsonName = "dimParam",
                        value = SimpleShardedDimProto::dimParam
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "num_shards",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                        jsonName = "numShards",
                        value = SimpleShardedDimProto::numShards
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class TrainingInfoProto(
    val initialization: GraphProto? = null,
    val algorithm: GraphProto? = null,
    val initializationBinding: List<StringStringEntryProto> = emptyList(),
    val updateBinding: List<StringStringEntryProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): TrainingInfoProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<TrainingInfoProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<TrainingInfoProto> {
        public val defaultInstance: TrainingInfoProto by lazy { TrainingInfoProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): TrainingInfoProto = TrainingInfoProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<TrainingInfoProto> = pbandk.MessageDescriptor(
            fullName = "TrainingInfoProto",
            messageClass = TrainingInfoProto::class,
            messageCompanion = this,
            fields = buildList(4) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "initialization",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = GraphProto.Companion),
                        jsonName = "initialization",
                        value = TrainingInfoProto::initialization
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "algorithm",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = GraphProto.Companion),
                        jsonName = "algorithm",
                        value = TrainingInfoProto::algorithm
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "initialization_binding",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "initializationBinding",
                        value = TrainingInfoProto::initializationBinding
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "update_binding",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "updateBinding",
                        value = TrainingInfoProto::updateBinding
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class ModelProto(
    val irVersion: Long = 0L,
    val opsetImport: List<OperatorSetIdProto> = emptyList(),
    val producerName: String = "",
    val producerVersion: String = "",
    val domain: String = "",
    val modelVersion: Long = 0L,
    val docString: String = "",
    val graph: GraphProto? = null,
    val metadataProps: List<StringStringEntryProto> = emptyList(),
    val trainingInfo: List<TrainingInfoProto> = emptyList(),
    val functions: List<FunctionProto> = emptyList(),
    val configuration: List<DeviceConfigurationProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): ModelProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<ModelProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<ModelProto> {
        public val defaultInstance: ModelProto by lazy { ModelProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): ModelProto = ModelProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<ModelProto> = pbandk.MessageDescriptor(
            fullName = "ModelProto",
            messageClass = ModelProto::class,
            messageCompanion = this,
            fields = buildList(12) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "ir_version",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                        jsonName = "irVersion",
                        value = ModelProto::irVersion
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "producer_name",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "producerName",
                        value = ModelProto::producerName
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "producer_version",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "producerVersion",
                        value = ModelProto::producerVersion
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "domain",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "domain",
                        value = ModelProto::domain
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "model_version",
                        number = 5,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                        jsonName = "modelVersion",
                        value = ModelProto::modelVersion
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "doc_string",
                        number = 6,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "docString",
                        value = ModelProto::docString
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "graph",
                        number = 7,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = GraphProto.Companion),
                        jsonName = "graph",
                        value = ModelProto::graph
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "opset_import",
                        number = 8,
                        type = pbandk.FieldDescriptor.Type.Repeated<OperatorSetIdProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = OperatorSetIdProto.Companion)),
                        jsonName = "opsetImport",
                        value = ModelProto::opsetImport
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "metadata_props",
                        number = 14,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "metadataProps",
                        value = ModelProto::metadataProps
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "training_info",
                        number = 20,
                        type = pbandk.FieldDescriptor.Type.Repeated<TrainingInfoProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = TrainingInfoProto.Companion)),
                        jsonName = "trainingInfo",
                        value = ModelProto::trainingInfo
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "functions",
                        number = 25,
                        type = pbandk.FieldDescriptor.Type.Repeated<FunctionProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = FunctionProto.Companion)),
                        jsonName = "functions",
                        value = ModelProto::functions
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "configuration",
                        number = 26,
                        type = pbandk.FieldDescriptor.Type.Repeated<DeviceConfigurationProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = DeviceConfigurationProto.Companion)),
                        jsonName = "configuration",
                        value = ModelProto::configuration
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class DeviceConfigurationProto(
    val name: String = "",
    val numDevices: Int = 0,
    val device: List<String> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): DeviceConfigurationProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<DeviceConfigurationProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<DeviceConfigurationProto> {
        public val defaultInstance: DeviceConfigurationProto by lazy { DeviceConfigurationProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): DeviceConfigurationProto = DeviceConfigurationProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<DeviceConfigurationProto> = pbandk.MessageDescriptor(
            fullName = "DeviceConfigurationProto",
            messageClass = DeviceConfigurationProto::class,
            messageCompanion = this,
            fields = buildList(3) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "name",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "name",
                        value = DeviceConfigurationProto::name
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "num_devices",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int32(),
                        jsonName = "numDevices",
                        value = DeviceConfigurationProto::numDevices
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "device",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Repeated<String>(valueType = pbandk.FieldDescriptor.Type.Primitive.String()),
                        jsonName = "device",
                        value = DeviceConfigurationProto::device
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class StringStringEntryProto(
    val key: String = "",
    val value: String = "",
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): StringStringEntryProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<StringStringEntryProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<StringStringEntryProto> {
        public val defaultInstance: StringStringEntryProto by lazy { StringStringEntryProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): StringStringEntryProto = StringStringEntryProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<StringStringEntryProto> = pbandk.MessageDescriptor(
            fullName = "StringStringEntryProto",
            messageClass = StringStringEntryProto::class,
            messageCompanion = this,
            fields = buildList(2) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "key",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "key",
                        value = StringStringEntryProto::key
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "value",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "value",
                        value = StringStringEntryProto::value
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class TensorAnnotation(
    val tensorName: String = "",
    val quantParameterTensorNames: List<StringStringEntryProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): TensorAnnotation = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<TensorAnnotation> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<TensorAnnotation> {
        public val defaultInstance: TensorAnnotation by lazy { TensorAnnotation() }
        override fun decodeWith(u: pbandk.MessageDecoder): TensorAnnotation = TensorAnnotation.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<TensorAnnotation> = pbandk.MessageDescriptor(
            fullName = "TensorAnnotation",
            messageClass = TensorAnnotation::class,
            messageCompanion = this,
            fields = buildList(2) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "tensor_name",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "tensorName",
                        value = TensorAnnotation::tensorName
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "quant_parameter_tensor_names",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "quantParameterTensorNames",
                        value = TensorAnnotation::quantParameterTensorNames
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class GraphProto(
    val node: List<NodeProto> = emptyList(),
    val name: String = "",
    val initializer: List<TensorProto> = emptyList(),
    val sparseInitializer: List<SparseTensorProto> = emptyList(),
    val docString: String = "",
    val input: List<ValueInfoProto> = emptyList(),
    val output: List<ValueInfoProto> = emptyList(),
    val valueInfo: List<ValueInfoProto> = emptyList(),
    val quantizationAnnotation: List<TensorAnnotation> = emptyList(),
    val metadataProps: List<StringStringEntryProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): GraphProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<GraphProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<GraphProto> {
        public val defaultInstance: GraphProto by lazy { GraphProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): GraphProto = GraphProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<GraphProto> = pbandk.MessageDescriptor(
            fullName = "GraphProto",
            messageClass = GraphProto::class,
            messageCompanion = this,
            fields = buildList(10) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "node",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Repeated<NodeProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = NodeProto.Companion)),
                        jsonName = "node",
                        value = GraphProto::node
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "name",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "name",
                        value = GraphProto::name
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "initializer",
                        number = 5,
                        type = pbandk.FieldDescriptor.Type.Repeated<TensorProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorProto.Companion)),
                        jsonName = "initializer",
                        value = GraphProto::initializer
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "doc_string",
                        number = 10,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "docString",
                        value = GraphProto::docString
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "input",
                        number = 11,
                        type = pbandk.FieldDescriptor.Type.Repeated<ValueInfoProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = ValueInfoProto.Companion)),
                        jsonName = "input",
                        value = GraphProto::input
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "output",
                        number = 12,
                        type = pbandk.FieldDescriptor.Type.Repeated<ValueInfoProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = ValueInfoProto.Companion)),
                        jsonName = "output",
                        value = GraphProto::output
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "value_info",
                        number = 13,
                        type = pbandk.FieldDescriptor.Type.Repeated<ValueInfoProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = ValueInfoProto.Companion)),
                        jsonName = "valueInfo",
                        value = GraphProto::valueInfo
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "quantization_annotation",
                        number = 14,
                        type = pbandk.FieldDescriptor.Type.Repeated<TensorAnnotation>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorAnnotation.Companion)),
                        jsonName = "quantizationAnnotation",
                        value = GraphProto::quantizationAnnotation
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "sparse_initializer",
                        number = 15,
                        type = pbandk.FieldDescriptor.Type.Repeated<SparseTensorProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = SparseTensorProto.Companion)),
                        jsonName = "sparseInitializer",
                        value = GraphProto::sparseInitializer
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "metadata_props",
                        number = 16,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "metadataProps",
                        value = GraphProto::metadataProps
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class TensorProto(
    val dims: List<Long> = emptyList(),
    val dataType: Int = 0,
    val segment: TensorProto.Segment? = null,
    val floatData: List<Float> = emptyList(),
    val int32Data: List<Int> = emptyList(),
    val stringData: List<pbandk.ByteArr> = emptyList(),
    val int64Data: List<Long> = emptyList(),
    val name: String = "",
    val docString: String = "",
    val rawData: pbandk.ByteArr = pbandk.ByteArr.empty,
    val externalData: List<StringStringEntryProto> = emptyList(),
    val dataLocation: TensorProto.DataLocation = TensorProto.DataLocation.fromValue(0),
    val doubleData: List<Double> = emptyList(),
    val uint64Data: List<Long> = emptyList(),
    val metadataProps: List<StringStringEntryProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): TensorProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<TensorProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<TensorProto> {
        public val defaultInstance: TensorProto by lazy { TensorProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): TensorProto = TensorProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<TensorProto> = pbandk.MessageDescriptor(
            fullName = "TensorProto",
            messageClass = TensorProto::class,
            messageCompanion = this,
            fields = buildList(15) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "dims",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Repeated<Long>(valueType = pbandk.FieldDescriptor.Type.Primitive.Int64(), packed = true),
                        jsonName = "dims",
                        value = TensorProto::dims
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "data_type",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int32(),
                        jsonName = "dataType",
                        value = TensorProto::dataType
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "segment",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorProto.Segment.Companion),
                        jsonName = "segment",
                        value = TensorProto::segment
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "float_data",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Repeated<Float>(valueType = pbandk.FieldDescriptor.Type.Primitive.Float(), packed = true),
                        jsonName = "floatData",
                        value = TensorProto::floatData
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "int32_data",
                        number = 5,
                        type = pbandk.FieldDescriptor.Type.Repeated<Int>(valueType = pbandk.FieldDescriptor.Type.Primitive.Int32(), packed = true),
                        jsonName = "int32Data",
                        value = TensorProto::int32Data
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "string_data",
                        number = 6,
                        type = pbandk.FieldDescriptor.Type.Repeated<pbandk.ByteArr>(valueType = pbandk.FieldDescriptor.Type.Primitive.Bytes()),
                        jsonName = "stringData",
                        value = TensorProto::stringData
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "int64_data",
                        number = 7,
                        type = pbandk.FieldDescriptor.Type.Repeated<Long>(valueType = pbandk.FieldDescriptor.Type.Primitive.Int64(), packed = true),
                        jsonName = "int64Data",
                        value = TensorProto::int64Data
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "name",
                        number = 8,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "name",
                        value = TensorProto::name
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "raw_data",
                        number = 9,
                        type = pbandk.FieldDescriptor.Type.Primitive.Bytes(),
                        jsonName = "rawData",
                        value = TensorProto::rawData
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "double_data",
                        number = 10,
                        type = pbandk.FieldDescriptor.Type.Repeated<Double>(valueType = pbandk.FieldDescriptor.Type.Primitive.Double(), packed = true),
                        jsonName = "doubleData",
                        value = TensorProto::doubleData
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "uint64_data",
                        number = 11,
                        type = pbandk.FieldDescriptor.Type.Repeated<Long>(valueType = pbandk.FieldDescriptor.Type.Primitive.UInt64(), packed = true),
                        jsonName = "uint64Data",
                        value = TensorProto::uint64Data
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "doc_string",
                        number = 12,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "docString",
                        value = TensorProto::docString
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "external_data",
                        number = 13,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "externalData",
                        value = TensorProto::externalData
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "data_location",
                        number = 14,
                        type = pbandk.FieldDescriptor.Type.Enum(enumCompanion = TensorProto.DataLocation.Companion),
                        jsonName = "dataLocation",
                        value = TensorProto::dataLocation
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "metadata_props",
                        number = 16,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "metadataProps",
                        value = TensorProto::metadataProps
                    )
                )
            }
        )
    }

    public sealed class DataType(override val value: Int, override val name: String? = null) : pbandk.Message.Enum {
        override fun equals(other: kotlin.Any?): Boolean = other is TensorProto.DataType && other.value == value
        override fun hashCode(): Int = value.hashCode()
        override fun toString(): String = "TensorProto.DataType.${name ?: "UNRECOGNIZED"}(value=$value)"

        public object UNDEFINED : DataType(0, "UNDEFINED")
        public object FLOAT : DataType(1, "FLOAT")
        public object UINT8 : DataType(2, "UINT8")
        public object INT8 : DataType(3, "INT8")
        public object UINT16 : DataType(4, "UINT16")
        public object INT16 : DataType(5, "INT16")
        public object INT32 : DataType(6, "INT32")
        public object INT64 : DataType(7, "INT64")
        public object STRING : DataType(8, "STRING")
        public object BOOL : DataType(9, "BOOL")
        public object FLOAT16 : DataType(10, "FLOAT16")
        public object DOUBLE : DataType(11, "DOUBLE")
        public object UINT32 : DataType(12, "UINT32")
        public object UINT64 : DataType(13, "UINT64")
        public object COMPLEX64 : DataType(14, "COMPLEX64")
        public object COMPLEX128 : DataType(15, "COMPLEX128")
        public object BFLOAT16 : DataType(16, "BFLOAT16")
        public object FLOAT8E4M3FN : DataType(17, "FLOAT8E4M3FN")
        public object FLOAT8E4M3FNUZ : DataType(18, "FLOAT8E4M3FNUZ")
        public object FLOAT8E5M2 : DataType(19, "FLOAT8E5M2")
        public object FLOAT8E5M2FNUZ : DataType(20, "FLOAT8E5M2FNUZ")
        public object UINT4 : DataType(21, "UINT4")
        public object INT4 : DataType(22, "INT4")
        public object FLOAT4E2M1 : DataType(23, "FLOAT4E2M1")
        public object FLOAT8E8M0 : DataType(24, "FLOAT8E8M0")
        public object UINT2 : DataType(25, "UINT2")
        public object INT2 : DataType(26, "INT2")
        public class UNRECOGNIZED(value: Int) : DataType(value)

        public companion object : pbandk.Message.Enum.Companion<TensorProto.DataType> {
            public val values: List<TensorProto.DataType> by lazy { listOf(UNDEFINED, FLOAT, UINT8, INT8, UINT16, INT16, INT32, INT64, STRING, BOOL, FLOAT16, DOUBLE, UINT32, UINT64, COMPLEX64, COMPLEX128, BFLOAT16, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ, UINT4, INT4, FLOAT4E2M1, FLOAT8E8M0, UINT2, INT2) }
            override fun fromValue(value: Int): TensorProto.DataType = values.firstOrNull { it.value == value } ?: UNRECOGNIZED(value)
            override fun fromName(name: String): TensorProto.DataType = values.firstOrNull { it.name == name } ?: throw IllegalArgumentException("No DataType with name: $name")
        }
    }

    public sealed class DataLocation(override val value: Int, override val name: String? = null) : pbandk.Message.Enum {
        override fun equals(other: kotlin.Any?): Boolean = other is TensorProto.DataLocation && other.value == value
        override fun hashCode(): Int = value.hashCode()
        override fun toString(): String = "TensorProto.DataLocation.${name ?: "UNRECOGNIZED"}(value=$value)"

        public object DEFAULT : DataLocation(0, "DEFAULT")
        public object EXTERNAL : DataLocation(1, "EXTERNAL")
        public class UNRECOGNIZED(value: Int) : DataLocation(value)

        public companion object : pbandk.Message.Enum.Companion<TensorProto.DataLocation> {
            public val values: List<TensorProto.DataLocation> by lazy { listOf(DEFAULT, EXTERNAL) }
            override fun fromValue(value: Int): TensorProto.DataLocation = values.firstOrNull { it.value == value } ?: UNRECOGNIZED(value)
            override fun fromName(name: String): TensorProto.DataLocation = values.firstOrNull { it.name == name } ?: throw IllegalArgumentException("No DataLocation with name: $name")
        }
    }

    public data class Segment(
        val begin: Long = 0L,
        val end: Long = 0L,
        override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
    ) : pbandk.Message {
        override operator fun plus(other: pbandk.Message?): TensorProto.Segment = protoMergeImpl(other)
        override val descriptor: pbandk.MessageDescriptor<TensorProto.Segment> get() = Companion.descriptor
        override val protoSize: Int by lazy { super.protoSize }
        public companion object : pbandk.Message.Companion<TensorProto.Segment> {
            public val defaultInstance: TensorProto.Segment by lazy { TensorProto.Segment() }
            override fun decodeWith(u: pbandk.MessageDecoder): TensorProto.Segment = TensorProto.Segment.decodeWithImpl(u)

            override val descriptor: pbandk.MessageDescriptor<TensorProto.Segment> = pbandk.MessageDescriptor(
                fullName = "TensorProto.Segment",
                messageClass = TensorProto.Segment::class,
                messageCompanion = this,
                fields = buildList(2) {
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "begin",
                            number = 1,
                            type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                            jsonName = "begin",
                            value = TensorProto.Segment::begin
                        )
                    )
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "end",
                            number = 2,
                            type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                            jsonName = "end",
                            value = TensorProto.Segment::end
                        )
                    )
                }
            )
        }
    }
}

@pbandk.Export
public data class SparseTensorProto(
    val values: TensorProto? = null,
    val indices: TensorProto? = null,
    val dims: List<Long> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): SparseTensorProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<SparseTensorProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<SparseTensorProto> {
        public val defaultInstance: SparseTensorProto by lazy { SparseTensorProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): SparseTensorProto = SparseTensorProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<SparseTensorProto> = pbandk.MessageDescriptor(
            fullName = "SparseTensorProto",
            messageClass = SparseTensorProto::class,
            messageCompanion = this,
            fields = buildList(3) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "values",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorProto.Companion),
                        jsonName = "values",
                        value = SparseTensorProto::values
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "indices",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorProto.Companion),
                        jsonName = "indices",
                        value = SparseTensorProto::indices
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "dims",
                        number = 3,
                        type = pbandk.FieldDescriptor.Type.Repeated<Long>(valueType = pbandk.FieldDescriptor.Type.Primitive.Int64(), packed = true),
                        jsonName = "dims",
                        value = SparseTensorProto::dims
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class TensorShapeProto(
    val dim: List<TensorShapeProto.Dimension> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): TensorShapeProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<TensorShapeProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<TensorShapeProto> {
        public val defaultInstance: TensorShapeProto by lazy { TensorShapeProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): TensorShapeProto = TensorShapeProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<TensorShapeProto> = pbandk.MessageDescriptor(
            fullName = "TensorShapeProto",
            messageClass = TensorShapeProto::class,
            messageCompanion = this,
            fields = buildList(1) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "dim",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Repeated<TensorShapeProto.Dimension>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorShapeProto.Dimension.Companion)),
                        jsonName = "dim",
                        value = TensorShapeProto::dim
                    )
                )
            }
        )
    }

    public data class Dimension(
        val denotation: String = "",
        val value: Value<*>? = null,
        override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
    ) : pbandk.Message {
        public sealed class Value<V>(value: V) : pbandk.Message.OneOf<V>(value) {
            public class DimValue(dimValue: Long = 0L) : Value<Long>(dimValue)
            public class DimParam(dimParam: String = "") : Value<String>(dimParam)
        }

        val dimValue: Long?
            get() = (value as? Value.DimValue)?.value
        val dimParam: String?
            get() = (value as? Value.DimParam)?.value

        override operator fun plus(other: pbandk.Message?): TensorShapeProto.Dimension = protoMergeImpl(other)
        override val descriptor: pbandk.MessageDescriptor<TensorShapeProto.Dimension> get() = Companion.descriptor
        override val protoSize: Int by lazy { super.protoSize }
        public companion object : pbandk.Message.Companion<TensorShapeProto.Dimension> {
            public val defaultInstance: TensorShapeProto.Dimension by lazy { TensorShapeProto.Dimension() }
            override fun decodeWith(u: pbandk.MessageDecoder): TensorShapeProto.Dimension = TensorShapeProto.Dimension.decodeWithImpl(u)

            override val descriptor: pbandk.MessageDescriptor<TensorShapeProto.Dimension> = pbandk.MessageDescriptor(
                fullName = "TensorShapeProto.Dimension",
                messageClass = TensorShapeProto.Dimension::class,
                messageCompanion = this,
                fields = buildList(3) {
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "dim_value",
                            number = 1,
                            type = pbandk.FieldDescriptor.Type.Primitive.Int64(hasPresence = true),
                            oneofMember = true,
                            jsonName = "dimValue",
                            value = TensorShapeProto.Dimension::dimValue
                        )
                    )
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "dim_param",
                            number = 2,
                            type = pbandk.FieldDescriptor.Type.Primitive.String(hasPresence = true),
                            oneofMember = true,
                            jsonName = "dimParam",
                            value = TensorShapeProto.Dimension::dimParam
                        )
                    )
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "denotation",
                            number = 3,
                            type = pbandk.FieldDescriptor.Type.Primitive.String(),
                            jsonName = "denotation",
                            value = TensorShapeProto.Dimension::denotation
                        )
                    )
                }
            )
        }
    }
}

@pbandk.Export
public data class TypeProto(
    val denotation: String = "",
    val value: Value<*>? = null,
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    public sealed class Value<V>(value: V) : pbandk.Message.OneOf<V>(value) {
        public class TensorType(tensorType: TypeProto.Tensor) : Value<TypeProto.Tensor>(tensorType)
        public class SequenceType(sequenceType: TypeProto.Sequence) : Value<TypeProto.Sequence>(sequenceType)
        public class MapType(mapType: TypeProto.Map_) : Value<TypeProto.Map_>(mapType)
        public class OptionalType(optionalType: TypeProto.Optional) : Value<TypeProto.Optional>(optionalType)
        public class SparseTensorType(sparseTensorType: TypeProto.SparseTensor) : Value<TypeProto.SparseTensor>(sparseTensorType)
        public class OpaqueType(opaqueType: TypeProto.Opaque) : Value<TypeProto.Opaque>(opaqueType)
    }

    val tensorType: TypeProto.Tensor?
        get() = (value as? Value.TensorType)?.value
    val sequenceType: TypeProto.Sequence?
        get() = (value as? Value.SequenceType)?.value
    val mapType: TypeProto.Map_?
        get() = (value as? Value.MapType)?.value
    val optionalType: TypeProto.Optional?
        get() = (value as? Value.OptionalType)?.value
    val sparseTensorType: TypeProto.SparseTensor?
        get() = (value as? Value.SparseTensorType)?.value
    val opaqueType: TypeProto.Opaque?
        get() = (value as? Value.OpaqueType)?.value

    override operator fun plus(other: pbandk.Message?): TypeProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<TypeProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<TypeProto> {
        public val defaultInstance: TypeProto by lazy { TypeProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): TypeProto = TypeProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<TypeProto> = pbandk.MessageDescriptor(
            fullName = "TypeProto",
            messageClass = TypeProto::class,
            messageCompanion = this,
            fields = buildList(7) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "tensor_type",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Tensor.Companion),
                        oneofMember = true,
                        jsonName = "tensorType",
                        value = TypeProto::tensorType
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "sequence_type",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Sequence.Companion),
                        oneofMember = true,
                        jsonName = "sequenceType",
                        value = TypeProto::sequenceType
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "map_type",
                        number = 5,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = Map_.Companion),
                        oneofMember = true,
                        jsonName = "mapType",
                        value = TypeProto::mapType
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "denotation",
                        number = 6,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "denotation",
                        value = TypeProto::denotation
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "opaque_type",
                        number = 7,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Opaque.Companion),
                        oneofMember = true,
                        jsonName = "opaqueType",
                        value = TypeProto::opaqueType
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "sparse_tensor_type",
                        number = 8,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.SparseTensor.Companion),
                        oneofMember = true,
                        jsonName = "sparseTensorType",
                        value = TypeProto::sparseTensorType
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "optional_type",
                        number = 9,
                        type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Optional.Companion),
                        oneofMember = true,
                        jsonName = "optionalType",
                        value = TypeProto::optionalType
                    )
                )
            }
        )
    }

    public data class Tensor(
        val elemType: Int = 0,
        val shape: TensorShapeProto? = null,
        override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
    ) : pbandk.Message {
        override operator fun plus(other: pbandk.Message?): Tensor = protoMergeImpl(other)
        override val descriptor: pbandk.MessageDescriptor<TypeProto.Tensor> get() = Companion.descriptor
        override val protoSize: Int by lazy { super.protoSize }
        public companion object : pbandk.Message.Companion<TypeProto.Tensor> {
            public val defaultInstance: TypeProto.Tensor by lazy { TypeProto.Tensor() }
            override fun decodeWith(u: pbandk.MessageDecoder): TypeProto.Tensor = TypeProto.Tensor.decodeWithImpl(u)

            override val descriptor: pbandk.MessageDescriptor<TypeProto.Tensor> = pbandk.MessageDescriptor(
                fullName = "TypeProto.Tensor",
                messageClass = TypeProto.Tensor::class,
                messageCompanion = this,
                fields = buildList(2) {
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "elem_type",
                            number = 1,
                            type = pbandk.FieldDescriptor.Type.Primitive.Int32(),
                            jsonName = "elemType",
                            value = TypeProto.Tensor::elemType
                        )
                    )
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "shape",
                            number = 2,
                            type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorShapeProto.Companion),
                            jsonName = "shape",
                            value = TypeProto.Tensor::shape
                        )
                    )
                }
            )
        }
    }

    public data class Sequence(
        val elemType: TypeProto? = null,
        override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
    ) : pbandk.Message {
        override operator fun plus(other: pbandk.Message?): TypeProto.Sequence = protoMergeImpl(other)
        override val descriptor: pbandk.MessageDescriptor<TypeProto.Sequence> get() = Companion.descriptor
        override val protoSize: Int by lazy { super.protoSize }
        public companion object : pbandk.Message.Companion<TypeProto.Sequence> {
            public val defaultInstance: TypeProto.Sequence by lazy { TypeProto.Sequence() }
            override fun decodeWith(u: pbandk.MessageDecoder): TypeProto.Sequence = TypeProto.Sequence.decodeWithImpl(u)

            override val descriptor: pbandk.MessageDescriptor<TypeProto.Sequence> = pbandk.MessageDescriptor(
                fullName = "TypeProto.Sequence",
                messageClass = TypeProto.Sequence::class,
                messageCompanion = this,
                fields = buildList(1) {
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "elem_type",
                            number = 1,
                            type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Companion),
                            jsonName = "elemType",
                            value = TypeProto.Sequence::elemType
                        )
                    )
                }
            )
        }
    }

    public data class Map_(
        val keyType: Int = 0,
        val valueType: TypeProto? = null,
        override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
    ) : pbandk.Message {
        override operator fun plus(other: pbandk.Message?): TypeProto.Map_ = protoMergeImpl(other)
        override val descriptor: pbandk.MessageDescriptor<TypeProto.Map_> get() = Companion.descriptor
        override val protoSize: Int by lazy { super.protoSize }
        public companion object : pbandk.Message.Companion<TypeProto.Map_> {
            public val defaultInstance: TypeProto.Map_ by lazy { TypeProto.Map_() }
            override fun decodeWith(u: pbandk.MessageDecoder): TypeProto.Map_ = TypeProto.Map_.decodeWithImpl(u)

            override val descriptor: pbandk.MessageDescriptor<TypeProto.Map_> = pbandk.MessageDescriptor(
                fullName = "TypeProto.Map",
                messageClass = TypeProto.Map_::class,
                messageCompanion = this,
                fields = buildList(2) {
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "key_type",
                            number = 1,
                            type = pbandk.FieldDescriptor.Type.Primitive.Int32(),
                            jsonName = "keyType",
                            value = TypeProto.Map_::keyType
                        )
                    )
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "value_type",
                            number = 2,
                            type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Companion),
                            jsonName = "valueType",
                            value = TypeProto.Map_::valueType
                        )
                    )
                }
            )
        }
    }

    public data class Optional(
        val elemType: TypeProto? = null,
        override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
    ) : pbandk.Message {
        override operator fun plus(other: pbandk.Message?): TypeProto.Optional = protoMergeImpl(other)
        override val descriptor: pbandk.MessageDescriptor<TypeProto.Optional> get() = Companion.descriptor
        override val protoSize: Int by lazy { super.protoSize }
        public companion object : pbandk.Message.Companion<TypeProto.Optional> {
            public val defaultInstance: TypeProto.Optional by lazy { TypeProto.Optional() }
            override fun decodeWith(u: pbandk.MessageDecoder): TypeProto.Optional = TypeProto.Optional.decodeWithImpl(u)

            override val descriptor: pbandk.MessageDescriptor<TypeProto.Optional> = pbandk.MessageDescriptor(
                fullName = "TypeProto.Optional",
                messageClass = TypeProto.Optional::class,
                messageCompanion = this,
                fields = buildList(1) {
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "elem_type",
                            number = 1,
                            type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TypeProto.Companion),
                            jsonName = "elemType",
                            value = TypeProto.Optional::elemType
                        )
                    )
                }
            )
        }
    }

    public data class SparseTensor(
        val elemType: Int = 0,
        val shape: TensorShapeProto? = null,
        override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
    ) : pbandk.Message {
        override operator fun plus(other: pbandk.Message?): TypeProto.SparseTensor = protoMergeImpl(other)
        override val descriptor: pbandk.MessageDescriptor<TypeProto.SparseTensor> get() = Companion.descriptor
        override val protoSize: Int by lazy { super.protoSize }
        public companion object : pbandk.Message.Companion<TypeProto.SparseTensor> {
            public val defaultInstance: TypeProto.SparseTensor by lazy { TypeProto.SparseTensor() }
            override fun decodeWith(u: pbandk.MessageDecoder): TypeProto.SparseTensor = TypeProto.SparseTensor.decodeWithImpl(u)

            override val descriptor: pbandk.MessageDescriptor<TypeProto.SparseTensor> = pbandk.MessageDescriptor(
                fullName = "TypeProto.SparseTensor",
                messageClass = TypeProto.SparseTensor::class,
                messageCompanion = this,
                fields = buildList(2) {
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "elem_type",
                            number = 1,
                            type = pbandk.FieldDescriptor.Type.Primitive.Int32(),
                            jsonName = "elemType",
                            value = TypeProto.SparseTensor::elemType
                        )
                    )
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "shape",
                            number = 2,
                            type = pbandk.FieldDescriptor.Type.Message(messageCompanion = TensorShapeProto.Companion),
                            jsonName = "shape",
                            value = TypeProto.SparseTensor::shape
                        )
                    )
                }
            )
        }
    }

    public data class Opaque(
        val domain: String = "",
        val name: String = "",
        override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
    ) : pbandk.Message {
        override operator fun plus(other: pbandk.Message?): TypeProto.Opaque = protoMergeImpl(other)
        override val descriptor: pbandk.MessageDescriptor<TypeProto.Opaque> get() = Companion.descriptor
        override val protoSize: Int by lazy { super.protoSize }
        public companion object : pbandk.Message.Companion<TypeProto.Opaque> {
            public val defaultInstance: TypeProto.Opaque by lazy { TypeProto.Opaque() }
            override fun decodeWith(u: pbandk.MessageDecoder): TypeProto.Opaque = TypeProto.Opaque.decodeWithImpl(u)

            override val descriptor: pbandk.MessageDescriptor<TypeProto.Opaque> = pbandk.MessageDescriptor(
                fullName = "TypeProto.Opaque",
                messageClass = TypeProto.Opaque::class,
                messageCompanion = this,
                fields = buildList(2) {
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "domain",
                            number = 1,
                            type = pbandk.FieldDescriptor.Type.Primitive.String(),
                            jsonName = "domain",
                            value = TypeProto.Opaque::domain
                        )
                    )
                    add(
                        pbandk.FieldDescriptor(
                            messageDescriptor = this@Companion::descriptor,
                            name = "name",
                            number = 2,
                            type = pbandk.FieldDescriptor.Type.Primitive.String(),
                            jsonName = "name",
                            value = TypeProto.Opaque::name
                        )
                    )
                }
            )
        }
    }
}

@pbandk.Export
public data class OperatorSetIdProto(
    val domain: String = "",
    val version: Long = 0L,
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): OperatorSetIdProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<OperatorSetIdProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<OperatorSetIdProto> {
        public val defaultInstance: OperatorSetIdProto by lazy { OperatorSetIdProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): OperatorSetIdProto = OperatorSetIdProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<OperatorSetIdProto> = pbandk.MessageDescriptor(
            fullName = "OperatorSetIdProto",
            messageClass = OperatorSetIdProto::class,
            messageCompanion = this,
            fields = buildList(2) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "domain",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "domain",
                        value = OperatorSetIdProto::domain
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "version",
                        number = 2,
                        type = pbandk.FieldDescriptor.Type.Primitive.Int64(),
                        jsonName = "version",
                        value = OperatorSetIdProto::version
                    )
                )
            }
        )
    }
}

@pbandk.Export
public data class FunctionProto(
    val name: String = "",
    val input: List<String> = emptyList(),
    val output: List<String> = emptyList(),
    val attribute: List<String> = emptyList(),
    val attributeProto: List<AttributeProto> = emptyList(),
    val node: List<NodeProto> = emptyList(),
    val docString: String = "",
    val opsetImport: List<OperatorSetIdProto> = emptyList(),
    val domain: String = "",
    val overload: String = "",
    val valueInfo: List<ValueInfoProto> = emptyList(),
    val metadataProps: List<StringStringEntryProto> = emptyList(),
    override val unknownFields: Map<Int, pbandk.UnknownField> = emptyMap()
) : pbandk.Message {
    override operator fun plus(other: pbandk.Message?): FunctionProto = protoMergeImpl(other)
    override val descriptor: pbandk.MessageDescriptor<FunctionProto> get() = Companion.descriptor
    override val protoSize: Int by lazy { super.protoSize }
    public companion object : pbandk.Message.Companion<FunctionProto> {
        public val defaultInstance: FunctionProto by lazy { FunctionProto() }
        override fun decodeWith(u: pbandk.MessageDecoder): FunctionProto = FunctionProto.decodeWithImpl(u)

        override val descriptor: pbandk.MessageDescriptor<FunctionProto> = pbandk.MessageDescriptor(
            fullName = "FunctionProto",
            messageClass = FunctionProto::class,
            messageCompanion = this,
            fields = buildList(12) {
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "name",
                        number = 1,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "name",
                        value = FunctionProto::name
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "input",
                        number = 4,
                        type = pbandk.FieldDescriptor.Type.Repeated<String>(valueType = pbandk.FieldDescriptor.Type.Primitive.String()),
                        jsonName = "input",
                        value = FunctionProto::input
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "output",
                        number = 5,
                        type = pbandk.FieldDescriptor.Type.Repeated<String>(valueType = pbandk.FieldDescriptor.Type.Primitive.String()),
                        jsonName = "output",
                        value = FunctionProto::output
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "attribute",
                        number = 6,
                        type = pbandk.FieldDescriptor.Type.Repeated<String>(valueType = pbandk.FieldDescriptor.Type.Primitive.String()),
                        jsonName = "attribute",
                        value = FunctionProto::attribute
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "node",
                        number = 7,
                        type = pbandk.FieldDescriptor.Type.Repeated<NodeProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = NodeProto.Companion)),
                        jsonName = "node",
                        value = FunctionProto::node
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "doc_string",
                        number = 8,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "docString",
                        value = FunctionProto::docString
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "opset_import",
                        number = 9,
                        type = pbandk.FieldDescriptor.Type.Repeated<OperatorSetIdProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = OperatorSetIdProto.Companion)),
                        jsonName = "opsetImport",
                        value = FunctionProto::opsetImport
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "domain",
                        number = 10,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "domain",
                        value = FunctionProto::domain
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "attribute_proto",
                        number = 11,
                        type = pbandk.FieldDescriptor.Type.Repeated<AttributeProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = AttributeProto.Companion)),
                        jsonName = "attributeProto",
                        value = FunctionProto::attributeProto
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "value_info",
                        number = 12,
                        type = pbandk.FieldDescriptor.Type.Repeated<ValueInfoProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = ValueInfoProto.Companion)),
                        jsonName = "valueInfo",
                        value = FunctionProto::valueInfo
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "overload",
                        number = 13,
                        type = pbandk.FieldDescriptor.Type.Primitive.String(),
                        jsonName = "overload",
                        value = FunctionProto::overload
                    )
                )
                add(
                    pbandk.FieldDescriptor(
                        messageDescriptor = this@Companion::descriptor,
                        name = "metadata_props",
                        number = 14,
                        type = pbandk.FieldDescriptor.Type.Repeated<StringStringEntryProto>(valueType = pbandk.FieldDescriptor.Type.Message(messageCompanion = StringStringEntryProto.Companion)),
                        jsonName = "metadataProps",
                        value = FunctionProto::metadataProps
                    )
                )
            }
        )
    }
}

@pbandk.Export
@pbandk.JsName("orDefaultForAttributeProto")
public fun AttributeProto?.orDefault(): AttributeProto = this ?: AttributeProto.defaultInstance

private fun AttributeProto.protoMergeImpl(plus: pbandk.Message?): AttributeProto = (plus as? AttributeProto)?.let {
    it.copy(
        t = t?.plus(plus.t) ?: plus.t,
        g = g?.plus(plus.g) ?: plus.g,
        sparseTensor = sparseTensor?.plus(plus.sparseTensor) ?: plus.sparseTensor,
        tp = tp?.plus(plus.tp) ?: plus.tp,
        floats = floats + plus.floats,
        ints = ints + plus.ints,
        strings = strings + plus.strings,
        tensors = tensors + plus.tensors,
        graphs = graphs + plus.graphs,
        sparseTensors = sparseTensors + plus.sparseTensors,
        typeProtos = typeProtos + plus.typeProtos,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun AttributeProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): AttributeProto {
    var name = ""
    var refAttrName = ""
    var docString = ""
    var type: AttributeProto.AttributeType = AttributeProto.AttributeType.fromValue(0)
    var f = 0.0F
    var i = 0L
    var s: pbandk.ByteArr = pbandk.ByteArr.empty
    var t: TensorProto? = null
    var g: GraphProto? = null
    var sparseTensor: SparseTensorProto? = null
    var tp: TypeProto? = null
    var floats: pbandk.ListWithSize.Builder<Float>? = null
    var ints: pbandk.ListWithSize.Builder<Long>? = null
    var strings: pbandk.ListWithSize.Builder<pbandk.ByteArr>? = null
    var tensors: pbandk.ListWithSize.Builder<TensorProto>? = null
    var graphs: pbandk.ListWithSize.Builder<GraphProto>? = null
    var sparseTensors: pbandk.ListWithSize.Builder<SparseTensorProto>? = null
    var typeProtos: pbandk.ListWithSize.Builder<TypeProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> name = _fieldValue as String
            2 -> f = _fieldValue as Float
            3 -> i = _fieldValue as Long
            4 -> s = _fieldValue as pbandk.ByteArr
            5 -> t = _fieldValue as TensorProto
            6 -> g = _fieldValue as GraphProto
            7 -> floats = (floats ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Float> }
            8 -> ints = (ints ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Long> }
            9 -> strings = (strings ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<pbandk.ByteArr> }
            10 -> tensors = (tensors ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<TensorProto> }
            11 -> graphs = (graphs ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<GraphProto> }
            13 -> docString = _fieldValue as String
            14 -> tp = _fieldValue as TypeProto
            15 -> typeProtos = (typeProtos ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<TypeProto> }
            20 -> type = _fieldValue as AttributeProto.AttributeType
            21 -> refAttrName = _fieldValue as String
            22 -> sparseTensor = _fieldValue as SparseTensorProto
            23 -> sparseTensors = (sparseTensors ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<SparseTensorProto> }
        }
    }

    return AttributeProto(name, refAttrName, docString, type,
        f, i, s, t,
        g, sparseTensor, tp, pbandk.ListWithSize.Builder.fixed(floats),
        pbandk.ListWithSize.Builder.fixed(ints), pbandk.ListWithSize.Builder.fixed(strings), pbandk.ListWithSize.Builder.fixed(tensors), pbandk.ListWithSize.Builder.fixed(graphs),
        pbandk.ListWithSize.Builder.fixed(sparseTensors), pbandk.ListWithSize.Builder.fixed(typeProtos), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForValueInfoProto")
public fun ValueInfoProto?.orDefault(): ValueInfoProto = this ?: ValueInfoProto.defaultInstance

private fun ValueInfoProto.protoMergeImpl(plus: pbandk.Message?): ValueInfoProto = (plus as? ValueInfoProto)?.let {
    it.copy(
        type = type?.plus(plus.type) ?: plus.type,
        metadataProps = metadataProps + plus.metadataProps,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun ValueInfoProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): ValueInfoProto {
    var name = ""
    var type: TypeProto? = null
    var docString = ""
    var metadataProps: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> name = _fieldValue as String
            2 -> type = _fieldValue as TypeProto
            3 -> docString = _fieldValue as String
            4 -> metadataProps = (metadataProps ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
        }
    }

    return ValueInfoProto(name, type, docString, pbandk.ListWithSize.Builder.fixed(metadataProps), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForNodeProto")
public fun NodeProto?.orDefault(): NodeProto = this ?: NodeProto.defaultInstance

private fun NodeProto.protoMergeImpl(plus: pbandk.Message?): NodeProto = (plus as? NodeProto)?.let {
    it.copy(
        input = input + plus.input,
        output = output + plus.output,
        attribute = attribute + plus.attribute,
        metadataProps = metadataProps + plus.metadataProps,
        deviceConfigurations = deviceConfigurations + plus.deviceConfigurations,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun NodeProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): NodeProto {
    var input: pbandk.ListWithSize.Builder<String>? = null
    var output: pbandk.ListWithSize.Builder<String>? = null
    var name = ""
    var opType = ""
    var domain = ""
    var overload = ""
    var attribute: pbandk.ListWithSize.Builder<AttributeProto>? = null
    var docString = ""
    var metadataProps: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null
    var deviceConfigurations: pbandk.ListWithSize.Builder<NodeDeviceConfigurationProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> input = (input ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<String> }
            2 -> output = (output ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<String> }
            3 -> name = _fieldValue as String
            4 -> opType = _fieldValue as String
            5 -> attribute = (attribute ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<AttributeProto> }
            6 -> docString = _fieldValue as String
            7 -> domain = _fieldValue as String
            8 -> overload = _fieldValue as String
            9 -> metadataProps = (metadataProps ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
            10 -> deviceConfigurations = (deviceConfigurations ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<NodeDeviceConfigurationProto> }
        }
    }

    return NodeProto(pbandk.ListWithSize.Builder.fixed(input), pbandk.ListWithSize.Builder.fixed(output), name, opType,
        domain, overload, pbandk.ListWithSize.Builder.fixed(attribute), docString,
        pbandk.ListWithSize.Builder.fixed(metadataProps), pbandk.ListWithSize.Builder.fixed(deviceConfigurations), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForIntIntListEntryProto")
public fun IntIntListEntryProto?.orDefault(): IntIntListEntryProto = this ?: IntIntListEntryProto.defaultInstance

private fun IntIntListEntryProto.protoMergeImpl(plus: pbandk.Message?): IntIntListEntryProto = (plus as? IntIntListEntryProto)?.let {
    it.copy(
        value = value + plus.value,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun IntIntListEntryProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): IntIntListEntryProto {
    var key = 0L
    var value: pbandk.ListWithSize.Builder<Long>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> key = _fieldValue as Long
            2 -> value = (value ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Long> }
        }
    }

    return IntIntListEntryProto(key, pbandk.ListWithSize.Builder.fixed(value), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForNodeDeviceConfigurationProto")
public fun NodeDeviceConfigurationProto?.orDefault(): NodeDeviceConfigurationProto = this ?: NodeDeviceConfigurationProto.defaultInstance

private fun NodeDeviceConfigurationProto.protoMergeImpl(plus: pbandk.Message?): NodeDeviceConfigurationProto = (plus as? NodeDeviceConfigurationProto)?.let {
    it.copy(
        shardingSpec = shardingSpec + plus.shardingSpec,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun NodeDeviceConfigurationProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): NodeDeviceConfigurationProto {
    var configurationId = ""
    var shardingSpec: pbandk.ListWithSize.Builder<ShardingSpecProto>? = null
    var pipelineStage = 0

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> configurationId = _fieldValue as String
            2 -> shardingSpec = (shardingSpec ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<ShardingSpecProto> }
            3 -> pipelineStage = _fieldValue as Int
        }
    }

    return NodeDeviceConfigurationProto(configurationId, pbandk.ListWithSize.Builder.fixed(shardingSpec), pipelineStage, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForShardingSpecProto")
public fun ShardingSpecProto?.orDefault(): ShardingSpecProto = this ?: ShardingSpecProto.defaultInstance

private fun ShardingSpecProto.protoMergeImpl(plus: pbandk.Message?): ShardingSpecProto = (plus as? ShardingSpecProto)?.let {
    it.copy(
        device = device + plus.device,
        indexToDeviceGroupMap = indexToDeviceGroupMap + plus.indexToDeviceGroupMap,
        shardedDim = shardedDim + plus.shardedDim,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun ShardingSpecProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): ShardingSpecProto {
    var tensorName = ""
    var device: pbandk.ListWithSize.Builder<Long>? = null
    var indexToDeviceGroupMap: pbandk.ListWithSize.Builder<IntIntListEntryProto>? = null
    var shardedDim: pbandk.ListWithSize.Builder<ShardedDimProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> tensorName = _fieldValue as String
            2 -> device = (device ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Long> }
            3 -> indexToDeviceGroupMap = (indexToDeviceGroupMap ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<IntIntListEntryProto> }
            4 -> shardedDim = (shardedDim ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<ShardedDimProto> }
        }
    }

    return ShardingSpecProto(tensorName, pbandk.ListWithSize.Builder.fixed(device), pbandk.ListWithSize.Builder.fixed(indexToDeviceGroupMap), pbandk.ListWithSize.Builder.fixed(shardedDim), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForShardedDimProto")
public fun ShardedDimProto?.orDefault(): ShardedDimProto = this ?: ShardedDimProto.defaultInstance

private fun ShardedDimProto.protoMergeImpl(plus: pbandk.Message?): ShardedDimProto = (plus as? ShardedDimProto)?.let {
    it.copy(
        simpleSharding = simpleSharding + plus.simpleSharding,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun ShardedDimProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): ShardedDimProto {
    var axis = 0L
    var simpleSharding: pbandk.ListWithSize.Builder<SimpleShardedDimProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> axis = _fieldValue as Long
            2 -> simpleSharding = (simpleSharding ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<SimpleShardedDimProto> }
        }
    }

    return ShardedDimProto(axis, pbandk.ListWithSize.Builder.fixed(simpleSharding), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForSimpleShardedDimProto")
public fun SimpleShardedDimProto?.orDefault(): SimpleShardedDimProto = this ?: SimpleShardedDimProto.defaultInstance

private fun SimpleShardedDimProto.protoMergeImpl(plus: pbandk.Message?): SimpleShardedDimProto = (plus as? SimpleShardedDimProto)?.let {
    it.copy(
        dim = plus.dim ?: dim,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun SimpleShardedDimProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): SimpleShardedDimProto {
    var numShards = 0L
    var dim: SimpleShardedDimProto.Dim<*>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> dim = SimpleShardedDimProto.Dim.DimValue(_fieldValue as Long)
            2 -> dim = SimpleShardedDimProto.Dim.DimParam(_fieldValue as String)
            3 -> numShards = _fieldValue as Long
        }
    }

    return SimpleShardedDimProto(numShards, dim, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTrainingInfoProto")
public fun TrainingInfoProto?.orDefault(): TrainingInfoProto = this ?: TrainingInfoProto.defaultInstance

private fun TrainingInfoProto.protoMergeImpl(plus: pbandk.Message?): TrainingInfoProto = (plus as? TrainingInfoProto)?.let {
    it.copy(
        initialization = initialization?.plus(plus.initialization) ?: plus.initialization,
        algorithm = algorithm?.plus(plus.algorithm) ?: plus.algorithm,
        initializationBinding = initializationBinding + plus.initializationBinding,
        updateBinding = updateBinding + plus.updateBinding,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TrainingInfoProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TrainingInfoProto {
    var initialization: GraphProto? = null
    var algorithm: GraphProto? = null
    var initializationBinding: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null
    var updateBinding: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> initialization = _fieldValue as GraphProto
            2 -> algorithm = _fieldValue as GraphProto
            3 -> initializationBinding = (initializationBinding ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
            4 -> updateBinding = (updateBinding ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
        }
    }

    return TrainingInfoProto(initialization, algorithm, pbandk.ListWithSize.Builder.fixed(initializationBinding), pbandk.ListWithSize.Builder.fixed(updateBinding), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForModelProto")
public fun ModelProto?.orDefault(): ModelProto = this ?: ModelProto.defaultInstance

private fun ModelProto.protoMergeImpl(plus: pbandk.Message?): ModelProto = (plus as? ModelProto)?.let {
    it.copy(
        opsetImport = opsetImport + plus.opsetImport,
        graph = graph?.plus(plus.graph) ?: plus.graph,
        metadataProps = metadataProps + plus.metadataProps,
        trainingInfo = trainingInfo + plus.trainingInfo,
        functions = functions + plus.functions,
        configuration = configuration + plus.configuration,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun ModelProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): ModelProto {
    var irVersion = 0L
    var opsetImport: pbandk.ListWithSize.Builder<OperatorSetIdProto>? = null
    var producerName = ""
    var producerVersion = ""
    var domain = ""
    var modelVersion = 0L
    var docString = ""
    var graph: GraphProto? = null
    var metadataProps: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null
    var trainingInfo: pbandk.ListWithSize.Builder<TrainingInfoProto>? = null
    var functions: pbandk.ListWithSize.Builder<FunctionProto>? = null
    var configuration: pbandk.ListWithSize.Builder<DeviceConfigurationProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> irVersion = _fieldValue as Long
            2 -> producerName = _fieldValue as String
            3 -> producerVersion = _fieldValue as String
            4 -> domain = _fieldValue as String
            5 -> modelVersion = _fieldValue as Long
            6 -> docString = _fieldValue as String
            7 -> graph = _fieldValue as GraphProto
            8 -> opsetImport = (opsetImport ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<OperatorSetIdProto> }
            14 -> metadataProps = (metadataProps ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
            20 -> trainingInfo = (trainingInfo ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<TrainingInfoProto> }
            25 -> functions = (functions ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<FunctionProto> }
            26 -> configuration = (configuration ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<DeviceConfigurationProto> }
        }
    }

    return ModelProto(irVersion, pbandk.ListWithSize.Builder.fixed(opsetImport), producerName, producerVersion,
        domain, modelVersion, docString, graph,
        pbandk.ListWithSize.Builder.fixed(metadataProps), pbandk.ListWithSize.Builder.fixed(trainingInfo), pbandk.ListWithSize.Builder.fixed(functions), pbandk.ListWithSize.Builder.fixed(configuration), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForDeviceConfigurationProto")
public fun DeviceConfigurationProto?.orDefault(): DeviceConfigurationProto = this ?: DeviceConfigurationProto.defaultInstance

private fun DeviceConfigurationProto.protoMergeImpl(plus: pbandk.Message?): DeviceConfigurationProto = (plus as? DeviceConfigurationProto)?.let {
    it.copy(
        device = device + plus.device,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun DeviceConfigurationProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): DeviceConfigurationProto {
    var name = ""
    var numDevices = 0
    var device: pbandk.ListWithSize.Builder<String>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> name = _fieldValue as String
            2 -> numDevices = _fieldValue as Int
            3 -> device = (device ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<String> }
        }
    }

    return DeviceConfigurationProto(name, numDevices, pbandk.ListWithSize.Builder.fixed(device), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForStringStringEntryProto")
public fun StringStringEntryProto?.orDefault(): StringStringEntryProto = this ?: StringStringEntryProto.defaultInstance

private fun StringStringEntryProto.protoMergeImpl(plus: pbandk.Message?): StringStringEntryProto = (plus as? StringStringEntryProto)?.let {
    it.copy(
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun StringStringEntryProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): StringStringEntryProto {
    var key = ""
    var value = ""

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> key = _fieldValue as String
            2 -> value = _fieldValue as String
        }
    }

    return StringStringEntryProto(key, value, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTensorAnnotation")
public fun TensorAnnotation?.orDefault(): TensorAnnotation = this ?: TensorAnnotation.defaultInstance

private fun TensorAnnotation.protoMergeImpl(plus: pbandk.Message?): TensorAnnotation = (plus as? TensorAnnotation)?.let {
    it.copy(
        quantParameterTensorNames = quantParameterTensorNames + plus.quantParameterTensorNames,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TensorAnnotation.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TensorAnnotation {
    var tensorName = ""
    var quantParameterTensorNames: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> tensorName = _fieldValue as String
            2 -> quantParameterTensorNames = (quantParameterTensorNames ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
        }
    }

    return TensorAnnotation(tensorName, pbandk.ListWithSize.Builder.fixed(quantParameterTensorNames), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForGraphProto")
public fun GraphProto?.orDefault(): GraphProto = this ?: GraphProto.defaultInstance

private fun GraphProto.protoMergeImpl(plus: pbandk.Message?): GraphProto = (plus as? GraphProto)?.let {
    it.copy(
        node = node + plus.node,
        initializer = initializer + plus.initializer,
        sparseInitializer = sparseInitializer + plus.sparseInitializer,
        input = input + plus.input,
        output = output + plus.output,
        valueInfo = valueInfo + plus.valueInfo,
        quantizationAnnotation = quantizationAnnotation + plus.quantizationAnnotation,
        metadataProps = metadataProps + plus.metadataProps,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun GraphProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): GraphProto {
    var node: pbandk.ListWithSize.Builder<NodeProto>? = null
    var name = ""
    var initializer: pbandk.ListWithSize.Builder<TensorProto>? = null
    var sparseInitializer: pbandk.ListWithSize.Builder<SparseTensorProto>? = null
    var docString = ""
    var input: pbandk.ListWithSize.Builder<ValueInfoProto>? = null
    var output: pbandk.ListWithSize.Builder<ValueInfoProto>? = null
    var valueInfo: pbandk.ListWithSize.Builder<ValueInfoProto>? = null
    var quantizationAnnotation: pbandk.ListWithSize.Builder<TensorAnnotation>? = null
    var metadataProps: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> node = (node ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<NodeProto> }
            2 -> name = _fieldValue as String
            5 -> initializer = (initializer ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<TensorProto> }
            10 -> docString = _fieldValue as String
            11 -> input = (input ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<ValueInfoProto> }
            12 -> output = (output ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<ValueInfoProto> }
            13 -> valueInfo = (valueInfo ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<ValueInfoProto> }
            14 -> quantizationAnnotation = (quantizationAnnotation ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<TensorAnnotation> }
            15 -> sparseInitializer = (sparseInitializer ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<SparseTensorProto> }
            16 -> metadataProps = (metadataProps ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
        }
    }

    return GraphProto(pbandk.ListWithSize.Builder.fixed(node), name, pbandk.ListWithSize.Builder.fixed(initializer), pbandk.ListWithSize.Builder.fixed(sparseInitializer),
        docString, pbandk.ListWithSize.Builder.fixed(input), pbandk.ListWithSize.Builder.fixed(output), pbandk.ListWithSize.Builder.fixed(valueInfo),
        pbandk.ListWithSize.Builder.fixed(quantizationAnnotation), pbandk.ListWithSize.Builder.fixed(metadataProps), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTensorProto")
public fun TensorProto?.orDefault(): TensorProto = this ?: TensorProto.defaultInstance

private fun TensorProto.protoMergeImpl(plus: pbandk.Message?): TensorProto = (plus as? TensorProto)?.let {
    it.copy(
        dims = dims + plus.dims,
        segment = segment?.plus(plus.segment) ?: plus.segment,
        floatData = floatData + plus.floatData,
        int32Data = int32Data + plus.int32Data,
        stringData = stringData + plus.stringData,
        int64Data = int64Data + plus.int64Data,
        externalData = externalData + plus.externalData,
        doubleData = doubleData + plus.doubleData,
        uint64Data = uint64Data + plus.uint64Data,
        metadataProps = metadataProps + plus.metadataProps,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TensorProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TensorProto {
    var dims: pbandk.ListWithSize.Builder<Long>? = null
    var dataType = 0
    var segment: TensorProto.Segment? = null
    var floatData: pbandk.ListWithSize.Builder<Float>? = null
    var int32Data: pbandk.ListWithSize.Builder<Int>? = null
    var stringData: pbandk.ListWithSize.Builder<pbandk.ByteArr>? = null
    var int64Data: pbandk.ListWithSize.Builder<Long>? = null
    var name = ""
    var docString = ""
    var rawData: pbandk.ByteArr = pbandk.ByteArr.empty
    var externalData: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null
    var dataLocation: TensorProto.DataLocation = TensorProto.DataLocation.fromValue(0)
    var doubleData: pbandk.ListWithSize.Builder<Double>? = null
    var uint64Data: pbandk.ListWithSize.Builder<Long>? = null
    var metadataProps: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> dims = (dims ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Long> }
            2 -> dataType = _fieldValue as Int
            3 -> segment = _fieldValue as TensorProto.Segment
            4 -> floatData = (floatData ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Float> }
            5 -> int32Data = (int32Data ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Int> }
            6 -> stringData = (stringData ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<pbandk.ByteArr> }
            7 -> int64Data = (int64Data ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Long> }
            8 -> name = _fieldValue as String
            9 -> rawData = _fieldValue as pbandk.ByteArr
            10 -> doubleData = (doubleData ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Double> }
            11 -> uint64Data = (uint64Data ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Long> }
            12 -> docString = _fieldValue as String
            13 -> externalData = (externalData ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
            14 -> dataLocation = _fieldValue as TensorProto.DataLocation
            16 -> metadataProps = (metadataProps ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
        }
    }

    return TensorProto(pbandk.ListWithSize.Builder.fixed(dims), dataType, segment, pbandk.ListWithSize.Builder.fixed(floatData),
        pbandk.ListWithSize.Builder.fixed(int32Data), pbandk.ListWithSize.Builder.fixed(stringData), pbandk.ListWithSize.Builder.fixed(int64Data), name,
        docString, rawData, pbandk.ListWithSize.Builder.fixed(externalData), dataLocation,
        pbandk.ListWithSize.Builder.fixed(doubleData), pbandk.ListWithSize.Builder.fixed(uint64Data), pbandk.ListWithSize.Builder.fixed(metadataProps), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTensorProtoSegment")
public fun TensorProto.Segment?.orDefault(): TensorProto.Segment = this ?: TensorProto.Segment.defaultInstance

private fun TensorProto.Segment.protoMergeImpl(plus: pbandk.Message?): TensorProto.Segment = (plus as? TensorProto.Segment)?.let {
    it.copy(
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TensorProto.Segment.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TensorProto.Segment {
    var begin = 0L
    var end = 0L

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> begin = _fieldValue as Long
            2 -> end = _fieldValue as Long
        }
    }

    return TensorProto.Segment(begin, end, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForSparseTensorProto")
public fun SparseTensorProto?.orDefault(): SparseTensorProto = this ?: SparseTensorProto.defaultInstance

private fun SparseTensorProto.protoMergeImpl(plus: pbandk.Message?): SparseTensorProto = (plus as? SparseTensorProto)?.let {
    it.copy(
        values = values?.plus(plus.values) ?: plus.values,
        indices = indices?.plus(plus.indices) ?: plus.indices,
        dims = dims + plus.dims,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun SparseTensorProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): SparseTensorProto {
    var values: TensorProto? = null
    var indices: TensorProto? = null
    var dims: pbandk.ListWithSize.Builder<Long>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> values = _fieldValue as TensorProto
            2 -> indices = _fieldValue as TensorProto
            3 -> dims = (dims ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<Long> }
        }
    }

    return SparseTensorProto(values, indices, pbandk.ListWithSize.Builder.fixed(dims), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTensorShapeProto")
public fun TensorShapeProto?.orDefault(): TensorShapeProto = this ?: TensorShapeProto.defaultInstance

private fun TensorShapeProto.protoMergeImpl(plus: pbandk.Message?): TensorShapeProto = (plus as? TensorShapeProto)?.let {
    it.copy(
        dim = dim + plus.dim,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TensorShapeProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TensorShapeProto {
    var dim: pbandk.ListWithSize.Builder<TensorShapeProto.Dimension>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> dim = (dim ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<TensorShapeProto.Dimension> }
        }
    }

    return TensorShapeProto(pbandk.ListWithSize.Builder.fixed(dim), unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTensorShapeProtoDimension")
public fun TensorShapeProto.Dimension?.orDefault(): TensorShapeProto.Dimension = this ?: TensorShapeProto.Dimension.defaultInstance

private fun TensorShapeProto.Dimension.protoMergeImpl(plus: pbandk.Message?): TensorShapeProto.Dimension = (plus as? TensorShapeProto.Dimension)?.let {
    it.copy(
        value = plus.value ?: value,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TensorShapeProto.Dimension.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TensorShapeProto.Dimension {
    var denotation = ""
    var value: TensorShapeProto.Dimension.Value<*>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> value = TensorShapeProto.Dimension.Value.DimValue(_fieldValue as Long)
            2 -> value = TensorShapeProto.Dimension.Value.DimParam(_fieldValue as String)
            3 -> denotation = _fieldValue as String
        }
    }

    return TensorShapeProto.Dimension(denotation, value, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTypeProto")
public fun TypeProto?.orDefault(): TypeProto = this ?: TypeProto.defaultInstance

private fun TypeProto.protoMergeImpl(plus: pbandk.Message?): TypeProto = (plus as? TypeProto)?.let {
    it.copy(
        value = when {
            value is TypeProto.Value.TensorType && plus.value is TypeProto.Value.TensorType ->
                TypeProto.Value.TensorType(value.value + plus.value.value)
            value is TypeProto.Value.SequenceType && plus.value is TypeProto.Value.SequenceType ->
                TypeProto.Value.SequenceType(value.value + plus.value.value)
            value is TypeProto.Value.MapType && plus.value is TypeProto.Value.MapType ->
                TypeProto.Value.MapType(value.value + plus.value.value)
            value is TypeProto.Value.OptionalType && plus.value is TypeProto.Value.OptionalType ->
                TypeProto.Value.OptionalType(value.value + plus.value.value)
            value is TypeProto.Value.SparseTensorType && plus.value is TypeProto.Value.SparseTensorType ->
                TypeProto.Value.SparseTensorType(value.value + plus.value.value)
            value is TypeProto.Value.OpaqueType && plus.value is TypeProto.Value.OpaqueType ->
                TypeProto.Value.OpaqueType(value.value + plus.value.value)
            else ->
                plus.value ?: value
        },
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TypeProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TypeProto {
    var denotation = ""
    var value: TypeProto.Value<*>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> value = TypeProto.Value.TensorType(_fieldValue as TypeProto.Tensor)
            4 -> value = TypeProto.Value.SequenceType(_fieldValue as TypeProto.Sequence)
            5 -> value = TypeProto.Value.MapType(_fieldValue as TypeProto.Map_)
            6 -> denotation = _fieldValue as String
            7 -> value = TypeProto.Value.OpaqueType(_fieldValue as TypeProto.Opaque)
            8 -> value = TypeProto.Value.SparseTensorType(_fieldValue as TypeProto.SparseTensor)
            9 -> value = TypeProto.Value.OptionalType(_fieldValue as TypeProto.Optional)
        }
    }

    return TypeProto(denotation, value, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTypeProtoTensor")
public fun TypeProto.Tensor?.orDefault(): TypeProto.Tensor = this ?: TypeProto.Tensor.defaultInstance

private fun TypeProto.Tensor.protoMergeImpl(plus: pbandk.Message?): TypeProto.Tensor = (plus as? TypeProto.Tensor)?.let {
    it.copy(
        shape = shape?.plus(plus.shape) ?: plus.shape,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TypeProto.Tensor.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TypeProto.Tensor {
    var elemType = 0
    var shape: TensorShapeProto? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> elemType = _fieldValue as Int
            2 -> shape = _fieldValue as TensorShapeProto
        }
    }

    return TypeProto.Tensor(elemType, shape, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTypeProtoSequence")
public fun TypeProto.Sequence?.orDefault(): TypeProto.Sequence = this ?: TypeProto.Sequence.defaultInstance

private fun TypeProto.Sequence.protoMergeImpl(plus: pbandk.Message?): TypeProto.Sequence = (plus as? TypeProto.Sequence)?.let {
    it.copy(
        elemType = elemType?.plus(plus.elemType) ?: plus.elemType,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TypeProto.Sequence.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TypeProto.Sequence {
    var elemType: TypeProto? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> elemType = _fieldValue as TypeProto
        }
    }

    return TypeProto.Sequence(elemType, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTypeProtoMap_")
public fun TypeProto.Map_?.orDefault(): TypeProto.Map_ = this ?: TypeProto.Map_.defaultInstance

private fun TypeProto.Map_.protoMergeImpl(plus: pbandk.Message?): TypeProto.Map_ = (plus as? TypeProto.Map_)?.let {
    it.copy(
        valueType = valueType?.plus(plus.valueType) ?: plus.valueType,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TypeProto.Map_.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TypeProto.Map_ {
    var keyType = 0
    var valueType: TypeProto? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> keyType = _fieldValue as Int
            2 -> valueType = _fieldValue as TypeProto
        }
    }

    return TypeProto.Map_(keyType, valueType, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTypeProtoOptional")
public fun TypeProto.Optional?.orDefault(): TypeProto.Optional = this ?: TypeProto.Optional.defaultInstance

private fun TypeProto.Optional.protoMergeImpl(plus: pbandk.Message?): TypeProto.Optional = (plus as? TypeProto.Optional)?.let {
    it.copy(
        elemType = elemType?.plus(plus.elemType) ?: plus.elemType,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TypeProto.Optional.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TypeProto.Optional {
    var elemType: TypeProto? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> elemType = _fieldValue as TypeProto
        }
    }

    return TypeProto.Optional(elemType, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTypeProtoSparseTensor")
public fun TypeProto.SparseTensor?.orDefault(): TypeProto.SparseTensor = this ?: TypeProto.SparseTensor.defaultInstance

private fun TypeProto.SparseTensor.protoMergeImpl(plus: pbandk.Message?): TypeProto.SparseTensor = (plus as? TypeProto.SparseTensor)?.let {
    it.copy(
        shape = shape?.plus(plus.shape) ?: plus.shape,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TypeProto.SparseTensor.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TypeProto.SparseTensor {
    var elemType = 0
    var shape: TensorShapeProto? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> elemType = _fieldValue as Int
            2 -> shape = _fieldValue as TensorShapeProto
        }
    }

    return TypeProto.SparseTensor(elemType, shape, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForTypeProtoOpaque")
public fun TypeProto.Opaque?.orDefault(): TypeProto.Opaque = this ?: TypeProto.Opaque.defaultInstance

private fun TypeProto.Opaque.protoMergeImpl(plus: pbandk.Message?): TypeProto.Opaque = (plus as? TypeProto.Opaque)?.let {
    it.copy(
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun TypeProto.Opaque.Companion.decodeWithImpl(u: pbandk.MessageDecoder): TypeProto.Opaque {
    var domain = ""
    var name = ""

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> domain = _fieldValue as String
            2 -> name = _fieldValue as String
        }
    }

    return TypeProto.Opaque(domain, name, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForOperatorSetIdProto")
public fun OperatorSetIdProto?.orDefault(): OperatorSetIdProto = this ?: OperatorSetIdProto.defaultInstance

private fun OperatorSetIdProto.protoMergeImpl(plus: pbandk.Message?): OperatorSetIdProto = (plus as? OperatorSetIdProto)?.let {
    it.copy(
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun OperatorSetIdProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): OperatorSetIdProto {
    var domain = ""
    var version = 0L

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> domain = _fieldValue as String
            2 -> version = _fieldValue as Long
        }
    }

    return OperatorSetIdProto(domain, version, unknownFields)
}

@pbandk.Export
@pbandk.JsName("orDefaultForFunctionProto")
public fun FunctionProto?.orDefault(): FunctionProto = this ?: FunctionProto.defaultInstance

private fun FunctionProto.protoMergeImpl(plus: pbandk.Message?): FunctionProto = (plus as? FunctionProto)?.let {
    it.copy(
        input = input + plus.input,
        output = output + plus.output,
        attribute = attribute + plus.attribute,
        attributeProto = attributeProto + plus.attributeProto,
        node = node + plus.node,
        opsetImport = opsetImport + plus.opsetImport,
        valueInfo = valueInfo + plus.valueInfo,
        metadataProps = metadataProps + plus.metadataProps,
        unknownFields = unknownFields + plus.unknownFields
    )
} ?: this

@Suppress("UNCHECKED_CAST")
private fun FunctionProto.Companion.decodeWithImpl(u: pbandk.MessageDecoder): FunctionProto {
    var name = ""
    var input: pbandk.ListWithSize.Builder<String>? = null
    var output: pbandk.ListWithSize.Builder<String>? = null
    var attribute: pbandk.ListWithSize.Builder<String>? = null
    var attributeProto: pbandk.ListWithSize.Builder<AttributeProto>? = null
    var node: pbandk.ListWithSize.Builder<NodeProto>? = null
    var docString = ""
    var opsetImport: pbandk.ListWithSize.Builder<OperatorSetIdProto>? = null
    var domain = ""
    var overload = ""
    var valueInfo: pbandk.ListWithSize.Builder<ValueInfoProto>? = null
    var metadataProps: pbandk.ListWithSize.Builder<StringStringEntryProto>? = null

    val unknownFields = u.readMessage(this) { _fieldNumber, _fieldValue ->
        when (_fieldNumber) {
            1 -> name = _fieldValue as String
            4 -> input = (input ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<String> }
            5 -> output = (output ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<String> }
            6 -> attribute = (attribute ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<String> }
            7 -> node = (node ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<NodeProto> }
            8 -> docString = _fieldValue as String
            9 -> opsetImport = (opsetImport ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<OperatorSetIdProto> }
            10 -> domain = _fieldValue as String
            11 -> attributeProto = (attributeProto ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<AttributeProto> }
            12 -> valueInfo = (valueInfo ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<ValueInfoProto> }
            13 -> overload = _fieldValue as String
            14 -> metadataProps = (metadataProps ?: pbandk.ListWithSize.Builder()).apply { this += _fieldValue as kotlin.sequences.Sequence<StringStringEntryProto> }
        }
    }

    return FunctionProto(name, pbandk.ListWithSize.Builder.fixed(input), pbandk.ListWithSize.Builder.fixed(output), pbandk.ListWithSize.Builder.fixed(attribute),
        pbandk.ListWithSize.Builder.fixed(attributeProto), pbandk.ListWithSize.Builder.fixed(node), docString, pbandk.ListWithSize.Builder.fixed(opsetImport),
        domain, overload, pbandk.ListWithSize.Builder.fixed(valueInfo), pbandk.ListWithSize.Builder.fixed(metadataProps), unknownFields)
}
