//
//  MLXChatModel.swift
//  MLXKit
//
//  Created by Aryan Rogye on 4/24/26.
//

import Foundation

/**
 * Struct represents a folder
 * with a id and a relativePath relative to the Documents URL
 */
public struct MLXChatModel: Identifiable, Hashable {
    public let id = UUID()
    var relativePath: String
    
    public var url: URL {
        URL.documentsDirectory.appendingPathComponent(relativePath)
    }
    
    public var name: String {
        url.lastPathComponent
    }
}
