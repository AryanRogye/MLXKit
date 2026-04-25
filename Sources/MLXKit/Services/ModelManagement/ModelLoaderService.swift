//
//  ModelLoaderService.swift
//  ModelChat
//
//  Created by Aryan Rogye on 4/5/26.
//

import Foundation
import SwiftUI

/**
 * Function Loads all the models and holds the selected model
 */
@Observable
@MainActor
public class ModelLoaderService {
    
    let modelDirectoryManager = ModelDirectoryManager()
    let downloadModelService : ModelDownloadService
    
    public var selected: MLXChatModel?
    public var models: [MLXChatModel] = []
    
    public var error: String?
    public var showError = false
    
    public var isSearching: Bool = false
    
    public var isDownloading : Bool {
        downloadModelService.isDownloading
    }
    
    public var status: String {
        downloadModelService.status
    }
    
    public var progress: Double {
        downloadModelService.progress
    }
    
    public var currentFileProgress: Double {
        downloadModelService.currentFileProgress
    }
    
    /// Pending States
    public var showContinueAlert = false
    public var pendingFiles: [String] = []
    private var downloadDecisionContinuation: CheckedContinuation<Bool, Never>?

    
    public init() {
        self.downloadModelService = ModelDownloadService(store: modelDirectoryManager)
        sync()
    }
    
    #if os(macOS)
    public func openModelFolder() {
        modelDirectoryManager.openBaseDirectory()
    }
    #endif
}

// MARK: - Sync
extension ModelLoaderService {
    /**
     * Sync all models
     */
    public func sync() {
        do {
            self.models = try modelDirectoryManager.getAllModelFolders()
        } catch {
            self.error = error.localizedDescription
            self.showError = true
        }
    }
}

// MARK: - Select
extension ModelLoaderService {
    /**
     * Function selects the model
     */
    public func select(
        _ model: MLXChatModel
    ) {
        if selected?.id == model.id {
            withAnimation(.bouncy) {
                selected = nil
            }
        } else {
            withAnimation(.bouncy) {
                selected = model
            }
        }
    }
}

// MARK: - Download
extension ModelLoaderService {
    /**
     * Download Name of model
     * function asks for confirmation before continuing with the download
     */
    public func download(
        named name: String
    ) async {
        do {
            try await downloadModelService.downloadModel(named: name) { [weak self] files in
                guard let self else { return false }
                return await self.askToContinue(with: files)
            }
            sync()
        } catch {
            self.error = error.localizedDescription
            self.showError = true
        }
    }
    
    /**
     * Function sets pendingFiles and
     * sets UI related things so the user can
     * select to continue
     */
    func askToContinue(
        with files: [String]
    ) async -> Bool {
        pendingFiles = files
        withAnimation(.bouncy) {
            showContinueAlert = true
        }
        
        return await withCheckedContinuation { continuation in
            downloadDecisionContinuation = continuation
        }
    }
    
    /**
     * Public API to download the model if (showContinueAlert)
     */
    public func userConfirmedDownload() {
        let continuation = downloadDecisionContinuation
        downloadDecisionContinuation = nil
        withAnimation(.bouncy) {
            showContinueAlert = false
        }
        continuation?.resume(returning: true)
    }
    
    /**
     * Public API to cancel downloading the model if (showContinueAlert)
     */
    public func userCancelledDownload() {
        let continuation = downloadDecisionContinuation
        downloadDecisionContinuation = nil
        withAnimation(.bouncy) {
            showContinueAlert = false
        }
        continuation?.resume(returning: false)
    }
}
