//
//  ModelDirectoryManager.swift
//  ModelChat
//
//  Created by Aryan Rogye on 4/5/26.
//

import Foundation
#if os(macOS)
import AppKit
#endif

enum ModelFolderStoreError: Error {
    case folderExists
}

/**
 * Ensures:
 * AppName/Documents/models/
 * all models gets loaded into this
 */
final class ModelDirectoryManager {
    let folderName = "models"
    
    init() {
        try? createBaseDirIfNeeded()
    }
    
    #if os(macOS)
    public func openBaseDirectory() {
        NSWorkspace.shared.open(URL.documentsDirectory
            .appendingPathComponent(folderName, isDirectory: true))
    }
    #endif
    
    /**
     * Helper to get all Folders
     */
    public func getAllModelFolders() throws -> [MLXChatModel] {
        try createBaseDirIfNeeded()
        
        let url : URL = URL.documentsDirectory
            .appendingPathComponent(folderName, isDirectory: true)
        
        return try FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        ).map { file in MLXChatModel(
            relativePath: "\(folderName)/\(file.lastPathComponent)"
        )}
    }
    
    /**
     * Function makes a new model folder in the models/ folder
     */
    public func makeModelFolder(
        named name: String
    ) throws -> URL {
        try createBaseDirIfNeeded()
        let url : URL = URL.documentsDirectory
            .appendingPathComponent(folderName, isDirectory: true)
            .appendingPathComponent(name, isDirectory: true)
        
        if directoryExists(at: url) {
            return url
        }
        
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true,
            attributes: nil
        )
        return url
    }
    
    /**
     * Function Creates the base directory
     * in this case is the App/Documents/models/
     */
    public func createBaseDirIfNeeded() throws {
        let folder = URL.documentsDirectory.appending(path: folderName)
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: folder.path,
            isDirectory: &isDirectory
        )
        
        if exists && !isDirectory.boolValue {
            /// delete it and make a folder
            try FileManager.default.removeItem(at: folder)
        }
        
        /// if it doesnt exist or it exists but its not a directory,
        /// this lets the top fall through
        if !exists || (exists && !isDirectory.boolValue) {
            // Create the folder if it didn't exist, or if we just deleted a conflicting file
            try FileManager.default.createDirectory(
                at: folder,
                withIntermediateDirectories: true,
                attributes: nil
            )
        }
    }
    
    
    /**
     * Internal function to verify if a directory exists
     * this is important because if is a file then its also
     * false
     */
    internal func directoryExists(at url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        if !FileManager.default.fileExists(
            atPath: url.path(),
            isDirectory: &isDirectory
        ) {
            /// this means it doesnt exist at all so just return false
            return false
        }
        return isDirectory.boolValue
    }
}
