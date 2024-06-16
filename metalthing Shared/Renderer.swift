//
//  Renderer.swift
//  metalthing Shared
//
//  Created by Johannes Loepelmann on 16.06.24.
//

// Our platform independent renderer class

import Metal
import MetalKit
import simd

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
}

class Renderer: NSObject, MTKViewDelegate {
    
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var offscreenPipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState
    var colorMap: MTLTexture
    var volumeTexture: MTLTexture
    var viewSize: CGSize
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var uniformBufferOffset = 0
    
    var uniformBufferIndex = 0
    
    var uniforms: UnsafeMutablePointer<Uniforms>
    
    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    
    var rotationMatrix: matrix_float4x4 = matrix_identity_float4x4
    var scaling: SIMD3<Float> = SIMD3<Float>(repeating: 4)
    
    var mesh: MTKMesh
    
    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight
        
        guard let buffer = self.device.makeBuffer(length:uniformBufferSize, options:[MTLResourceOptions.storageModeShared]) else { return nil }
        dynamicUniformBuffer = buffer
        
        self.dynamicUniformBuffer.label = "UniformBuffer"
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:Uniforms.self, capacity:1)
        
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1
        
        viewSize = metalKitView.drawableSize
        
        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()
        
        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDescriptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor:depthStateDescriptor) else { return nil }
        depthState = state
        
        do {
            mesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }
        
        guard let texture = Renderer.CreateOffscreenTexture(device: device, size: viewSize) else {
            print("Unable to create Texture")
            return nil
        }
        
        colorMap = texture
        
        do {
            offscreenPipelineState = try Renderer.buildOffscreenPipelineWithDevice(device: device, vertexDescriptor: mtlVertexDescriptor, target: colorMap)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        let res = Renderer.loadVolFile(device: device, name: "Frog")
        
        guard let vol = res.0 , let _scale = res.1 else {
            print("Unable to load Volume")
            return nil
        }
        
        volumeTexture = vol
        scaling = scaling * SIMD3<Float>(0.5, 1, 1)
        
        super.init()
    }
    
    class func CreateOffscreenTexture(device: MTLDevice, size: CGSize) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor()
        descriptor.width = Int(size.width)
        descriptor.height = Int(size.height)
        descriptor.pixelFormat = .bgra8Unorm_srgb
        descriptor.usage = [.renderTarget, .shaderRead]
        
        if let texture = device.makeTexture(descriptor: descriptor) {
            return texture
        }
        else {
            print("Unable to create Texture")
            return nil
        }
    }
    
    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices
        
        let mtlVertexDescriptor = MTLVertexDescriptor()
        
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue
        
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue
        
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        return mtlVertexDescriptor
    }
    
    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object
        
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.rasterSampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    class func buildOffscreenPipelineWithDevice(device: MTLDevice, vertexDescriptor: MTLVertexDescriptor, target: MTLTexture) throws -> MTLRenderPipelineState {
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "backfaceFragment")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.rasterSampleCount = 1
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        pipelineDescriptor.colorAttachments[0].pixelFormat = target.pixelFormat
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    class func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
        let mdlMesh = MDLMesh.newBox(withDimensions: SIMD3<Float>(1, 1, 1),
                                     segments: SIMD3<UInt32>(1, 1, 1),
                                     geometryType: MDLGeometryType.triangles,
                                     inwardNormals:false,
                                     allocator: metalAllocator)
        
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
        
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        
        mdlMesh.vertexDescriptor = mdlVertexDescriptor
        
        return try MTKMesh(mesh:mdlMesh, device:device)
    }
    
    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling
        
        let textureLoader = MTKTextureLoader(device: device)
        
        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]
        
        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
        
    }
    
    class func loadVolFile(device: MTLDevice, name: String) -> (MTLTexture?, SIMD3<Float>?) {
        guard let file = NSDataAsset(name: name)?.data
        else {
            print("Cant open \(name)");
            return (nil, nil)
        }
        
        let raw = [UInt8] (file)
        
        let width = Int(Renderer.BigEndianIntFromBytes(data: raw[0...3]))
        let height = Int(Renderer.BigEndianIntFromBytes(data: raw[4...7]))
        let depth = Int(Renderer.BigEndianIntFromBytes(data: raw[8...11]))
        // unused 32bit Int: 12-15
        // three 32 bit floats (scaling): 16-27
        let scaleX = Renderer.BigEndianFloatFromBytes(data: raw[16...19])
        let scaleY = Renderer.BigEndianFloatFromBytes(data: raw[20...23])
        let scaleZ = Renderer.BigEndianFloatFromBytes(data: raw[24...27])
        
        let scale = SIMD3<Float>(scaleX, scaleY, scaleZ)
        
        
        let dataStart = 28
        
        let descriptor = MTLTextureDescriptor()
        descriptor.width = width
        descriptor.height = height
        descriptor.depth = depth
        descriptor.pixelFormat = .r8Unorm
        descriptor.usage = [.shaderRead]
        descriptor.textureType = .type3D
        
        var cube = [UInt8](repeating: 0, count: width * height * depth)
        
        for x in 0...width-1 {
            for y in 0...height-1 {
                for z in 0...depth-1 {
                    let value = raw[(x * height * depth) + (y * depth) + z + dataStart]
                    cube[x + (y * width) + (z * width * height)] = value
                }
            }
        }
        
        if let texture = device.makeTexture(descriptor: descriptor) {
            
            texture.replace(region: MTLRegion.init(origin: MTLOrigin(x: 0,y: 0,z: 0), size: MTLSize(width: width, height: height, depth: depth)), mipmapLevel: 0, slice: 0, withBytes: cube, bytesPerRow: width, bytesPerImage: width*height)
            
            return (texture, scale)
        }
        else {
            print("Unable to create Texture")
            return (nil, nil)
        }
    }
    
    class func BigEndianIntFromBytes(data: ArraySlice<UInt8>) -> UInt32 {
        let start = data.startIndex
        
        let value =
        (UInt32(data[start + 0]) << (3*8)) |
        (UInt32(data[start + 1]) << (2*8)) |
        (UInt32(data[start + 2]) << (1*8)) |
        (UInt32(data[start + 3]) << (0*8))
        
        return value
    }
    
    class func BigEndianFloatFromBytes(data: ArraySlice<UInt8>) -> Float {
        let start = data.startIndex
        
        let value =
        (UInt32(data[start + 0]) << (3*8)) |
        (UInt32(data[start + 1]) << (2*8)) |
        (UInt32(data[start + 2]) << (1*8)) |
        (UInt32(data[start + 3]) << (0*8))
        
        return Float(bitPattern: value)
    }
    
    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
        
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
    }
    
    private func updateGameState() {
        /// Update any game state before rendering
        
        uniforms[0].projectionMatrix = projectionMatrix
        
        let modelMatrix = rotationMatrix * matrix4x4_scaling(factor: scaling)
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0)
        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
        
        uniforms[0].screenSize = SIMD2<Float>(Float(viewSize.width), Float(viewSize.height))
    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
            }
            
            self.updateDynamicBufferState()
            
            self.updateGameState()
            
            let offscreenPassDescriptor = MTLRenderPassDescriptor()
            offscreenPassDescriptor.colorAttachments[0].texture = colorMap
            offscreenPassDescriptor.colorAttachments[0].loadAction = .clear
            offscreenPassDescriptor.colorAttachments[0].clearColor = .init(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
            offscreenPassDescriptor.colorAttachments[0].storeAction = .store
            
            if let offsreenEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: offscreenPassDescriptor) {
                offsreenEncoder.label = "Backface Encoder"
                offsreenEncoder.setRenderPipelineState(offscreenPipelineState)
                offsreenEncoder.setCullMode(.front)
                offsreenEncoder.setFrontFacing(.counterClockwise)
                
                offsreenEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                offsreenEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
                    guard let layout = element as? MDLVertexBufferLayout else {
                        return
                    }
                    
                    if layout.stride != 0 {
                        let buffer = mesh.vertexBuffers[index]
                        offsreenEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
                    }
                }
                
                offsreenEncoder.setFragmentTexture(colorMap, index: TextureIndex.color.rawValue)
                
                for submesh in mesh.submeshes {
                    offsreenEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                          indexCount: submesh.indexCount,
                                                          indexType: submesh.indexType,
                                                          indexBuffer: submesh.indexBuffer.buffer,
                                                          indexBufferOffset: submesh.indexBuffer.offset)
                    
                }
                
                offsreenEncoder.endEncoding()
            }
            
            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor
            
            if let renderPassDescriptor = renderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                
                /// Final pass rendering code here
                renderEncoder.label = "Primary Render Encoder"
                
                renderEncoder.pushDebugGroup("Draw Frontface")
                
                renderEncoder.setCullMode(.back)
                
                renderEncoder.setFrontFacing(.counterClockwise)
                
                renderEncoder.setRenderPipelineState(pipelineState)
                
                renderEncoder.setDepthStencilState(depthState)
                
                renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
                    guard let layout = element as? MDLVertexBufferLayout else {
                        return
                    }
                    
                    if layout.stride != 0 {
                        let buffer = mesh.vertexBuffers[index]
                        renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
                    }
                }
                
                renderEncoder.setFragmentTexture(colorMap, index: TextureIndex.color.rawValue)
                renderEncoder.setFragmentTexture(volumeTexture, index: TextureIndex.volume.rawValue)
                
                for submesh in mesh.submeshes {
                    renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                        indexCount: submesh.indexCount,
                                                        indexType: submesh.indexType,
                                                        indexBuffer: submesh.indexBuffer.buffer,
                                                        indexBufferOffset: submesh.indexBuffer.offset)
                    
                }
                
                renderEncoder.popDebugGroup()
                renderEncoder.endEncoding()
                
                if let drawable = view.currentDrawable {
                    commandBuffer.present(drawable)
                }
            }
            
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here
        
        let aspect = Float(size.width) / Float(size.height)
        viewSize = size
        projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(65), aspectRatio:aspect, nearZ: 0.1, farZ: 100.0)
        
        guard let texture = Renderer.CreateOffscreenTexture(device: device, size: size) else {
            print("Failed to create texture")
            return;
        }
        
        colorMap = texture
    }
    
    func rotate(x: Float, y: Float) {
        let scaling: Float = 5.0
        if x != 0 {
            rotationMatrix = matrix4x4_rotation(radians: x * scaling, axis: SIMD3<Float>(0, 1, 0)) * rotationMatrix
        }
        if y != 0 {
            rotationMatrix = matrix4x4_rotation(radians: y * scaling, axis: SIMD3<Float>(1, 0, 0)) * rotationMatrix
        }
    }

}


// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_scaling(factor: SIMD3<Float>) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(factor.x,        0,        0, 0),
                                         vector_float4(0       , factor.y,        0, 0),
                                         vector_float4(0       ,        0, factor.z, 0),
                                         vector_float4(0       ,        0,        0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}
