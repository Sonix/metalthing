//
//  GameViewController.swift
//  metalthing iOS
//
//  Created by Johannes Loepelmann on 16.06.24.
//

import UIKit
import MetalKit

// Our iOS specific view controller
class GameViewController: UIViewController {

    var renderer: Renderer!
    var mtkView: MTKView!

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View of Gameview controller is not an MTKView")
            return
        }

        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported")
            return
        }

        mtkView.device = defaultDevice
        mtkView.backgroundColor = UIColor.black

        guard let newRenderer = Renderer(metalKitView: mtkView) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer

        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)

        mtkView.delegate = renderer
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        let touch = touches.first!
        let deltaX = (touch.location(in: mtkView).x - touch.previousLocation(in: mtkView).x) / self.view.frame.width
        let deltaY = (touch.location(in: mtkView).y - touch.previousLocation(in: mtkView).y) / self.view.frame.height
        
        renderer.rotate(x: Float(deltaX), y: Float(deltaY))
    }
}
