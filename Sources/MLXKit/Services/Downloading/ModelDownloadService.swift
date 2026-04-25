//
//  ModelDownloadService.swift
//  ModelChat
//
//  Created by Aryan Rogye on 4/5/26.
//

import Foundation
import Observation

/**
 * Internally Used
 * Users should rather be using the ModelLoaderService since it manages confirmations
 */
@MainActor
@Observable
final class ModelDownloadService {
    
    var isDownloading = false
    var status = ""
    var progress: Double = 0
    var currentFileProgress: Double = 0
    
    let store: ModelDirectoryManager
    private let downloader = FileDownloader()
    
    init(store: ModelDirectoryManager) {
        self.store = store
        downloader.onProgress = { [weak self] p in
            Task { @MainActor in
                self?.currentFileProgress = p
            }
        }
    }
    
    func downloadModel(
        named modelName: String,
        continueDownload: @escaping ([String]) async -> Bool
    ) async throws {
        isDownloading = true
        status = "Fetching file list..."
        progress = 0
        currentFileProgress = 0
        defer { isDownloading = false }
        
        let files = try await fetchFileList(for: modelName)
        if await !continueDownload(files) {
            return
        }
        let modelFolderURL = try store.makeModelFolder(named: modelName)
        print("Found Files: \(files)")

        let downloadable = files.filter { file in
            !file.hasPrefix(".") &&
            !file.hasSuffix(".md") &&
            file != "LICENSE"
        }
        

        for (index, file) in downloadable.enumerated() {
            let localURL = modelFolderURL.appendingPathComponent(file)
            
            // Skip if already downloaded and non-empty
            if let attrs = try? FileManager.default.attributesOfItem(atPath: localURL.path),
               let size = attrs[.size] as? Int, size > 0 {
                progress = Double(index + 1) / Double(downloadable.count)
                currentFileProgress = 1
                continue
            }
            
            status = "Downloading \(file) (\(index + 1)/\(downloadable.count))..."
            currentFileProgress = 0
            
            let remoteURL = try makeHuggingFaceURL(
                repo: "mlx-community/\(modelName)",
                fileName: file
            )
            
            try await downloader.download(from: remoteURL, to: localURL)
            
            progress = Double(index + 1) / Double(downloadable.count)
            currentFileProgress = 1
        }
        
        status = "Done"
        return
    }
    
    private func fetchFileList(for modelName: String) async throws -> [String] {
        guard let url = URL(string: "https://huggingface.co/api/models/mlx-community/\(modelName)") else {
            throw URLError(.badURL)
        }
        let (data, _) = try await URLSession.shared.data(from: url)
        let json = try JSONDecoder().decode(HFModelInfo.self, from: data)
        return json.siblings.map { $0.rfilename }
    }
    
    private struct HFModelInfo: Decodable {
        let siblings: [Sibling]
        struct Sibling: Decodable {
            let rfilename: String
        }
    }
    
    private func makeHuggingFaceURL(repo: String, fileName: String) throws -> URL {
        guard
            let encodedRepo = repo.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
            let encodedFile = fileName.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
            let url = URL(string: "https://huggingface.co/\(encodedRepo)/resolve/main/\(encodedFile)")
        else {
            throw URLError(.badURL)
        }
        return url
    }
}
